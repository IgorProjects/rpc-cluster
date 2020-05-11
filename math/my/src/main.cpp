#pragma warning(disable:4996)
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <omp.h>

#include <random>
#include <chrono>
#include <boost/container/static_vector.hpp>
#include <thread>

namespace mpi = boost::mpi;
namespace ublas = boost::numeric::ublas;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
constexpr int MakeMessageTag(const char* tag)
{
    auto result = 0, i = 0;
    for (; i < 4 && *tag != 0; ++i, ++tag)
    {
        result |= *tag << 8 * i;
    }
    assert(*tag == 0);
    for (; i < 4; ++i)
    {
        result |= '_' << 8 * i;
    }
    return result;
}

double f(size_t i, size_t j, size_t n);
double u(size_t i, size_t j, size_t n);

void Print(const ublas::matrix<double>& matrix)
{
    for (auto i = 0; i < matrix.size1(); ++i)
    {
        for (auto j = 0; j < matrix.size2(); ++j)
        {
            std::cout << matrix(i, j) << ' ';
        }
        std::cout << std::endl;
    }
}

void GausSeidelIteration(
    ublas::matrix_range<ublas::matrix<double>> matrix, 
    size_t i, size_t j, double h, 
    std::vector<double>& deltas)
{
    auto temp = matrix(i, j);
    matrix(i, j) = (matrix(i - 1, j) + matrix(i + 1, j) +
        matrix(i, j - 1) + matrix(i, j + 1) - h * h * f(i, j, matrix.size1() - 2)) / 4;
    auto delta = fabs(temp - matrix(i, j));
    if (deltas[i - 1] < delta)
    {
        deltas[i - 1] = delta;
    }
}

double GaussSeidel(ublas::matrix_range<ublas::matrix<double>> matrix, double h)
{
    const auto maxWaveSize = matrix.size1() - 2;
    std::vector<double> deltas(maxWaveSize);
    for (auto waveSize = 1; waveSize < maxWaveSize + 1; ++waveSize) 
    {
#pragma omp parallel for shared(matrix,waveSize,deltas)
        for (int i = 1; i < waveSize + 1; ++i) 
        {
            auto j = waveSize + 1 - i;
            GausSeidelIteration(matrix, i, j, h, deltas);
        }
    }
    for (int waveSize = maxWaveSize - 1; waveSize > 0; --waveSize) 
    {
#pragma omp parallel for shared(matrix,waveSize,deltas)
        for (int i = maxWaveSize - waveSize + 1; i < maxWaveSize + 1; ++i) 
        {
            auto j = 2 * maxWaveSize - waveSize - i + 1;
            GausSeidelIteration(matrix, i, j, h, deltas);
        }
    }
    return *std::max_element(deltas.begin(), deltas.end());
}

ublas::vector<double> ReceiveVector(const mpi::communicator& communicator, const int source, const size_t size, const char* tag);

int main(int argc, char* argv[])
{
    mpi::environment env;
    mpi::communicator world;
    const auto gridSize = world.size() - 1;

    ublas::matrix<double> matrix;
    double eps, h;
    size_t n, blockSize;
    if (world.rank() == 0)
    {
        if (argc < 3)
        {
            std::cout << "Wrong parameters" << std::endl;
            return 1;
        }
        char* end;
        n = strtoul(argv[1], &end, 10); // size of grid
        eps = strtod(argv[2], &end);
        if ((n - 2) % gridSize != 0)
        {
            std::cout << "Wrong size" << std::endl;
            return 2;
        }
        blockSize = (n - 2) / gridSize;
        h = 1 / static_cast<double>(n - 1);
    }
    mpi::broadcast(world, blockSize, 0);
    mpi::broadcast(world, eps, 0);
    mpi::broadcast(world, h, 0);
    mpi::broadcast(world, n, 0);
    matrix.resize(n, n);
    if (world.rank() == 0)
    {
        for (auto i = 0; i < n; ++i)
        {
            for (auto j = 0; j < n; ++j)
            {
                matrix(i, j) = u(i, j, n);
            }
        }
    }
    mpi::broadcast(world, matrix, 0);
    double procDelta = 0, delta = 0;
    const auto currentTime = std::chrono::high_resolution_clock::now();
    do
    {
        
        if (world.rank() != 0)
        {
            procDelta = 0;
            auto j = 0, i = gridSize - world.rank();
            double iterationDelta;
            boost::container::static_vector<mpi::request, 3> requests;
            ublas::vector<double> tempVector1{blockSize}, tempVector2{blockSize};
            // row
            for (; j < world.rank() - 1; ++j)
            {
                auto currentBlock = 
                    ublas::subrange(matrix,
                        i * blockSize, (i + 1) * blockSize + 2,
                        j * blockSize + 1, (j + 1) * blockSize + 1);
                requests.clear();
                if (i != 0)
                {
                    // send old top row
                    requests.push_back(
                        world.isend(world.rank() + 1, MakeMessageTag("orow"), 
                            ublas::vector<double>{ublas::row(currentBlock, 1)}));
                    // receive top row
                    requests.push_back(
                        world.irecv(world.rank() + 1, MakeMessageTag("nrow"), tempVector1));
                }
                // receive old bottom row
                requests.push_back(
                    world.irecv(world.rank() - 1, MakeMessageTag("orow"), tempVector2));
                mpi::wait_all(requests.begin(), requests.end());
                if (i != 0)
                {
                    ublas::row(currentBlock, 0) = tempVector1;
                }
                ublas::row(currentBlock, currentBlock.size1() - 1) = tempVector2;
                iterationDelta = GaussSeidel(ublas::subrange(matrix,
                    i * blockSize, (i + 1) * blockSize + 2,
                    j * blockSize, (j + 1) * blockSize + 2), h);
                if (iterationDelta > procDelta)
                {
                    procDelta = iterationDelta;
                }
                if (i != gridSize - 1)
                {
                    // send bottom row
                    mpi::request request =
                        world.isend(world.rank() - 1, MakeMessageTag("nrow"),
                            ublas::vector<double>{ublas::row(currentBlock, currentBlock.size1() - 2)});
                    request.wait();
                }
            }
            // angle
            {
                auto currentBlockRow = 
                    ublas::subrange(matrix,
                        i * blockSize, (i + 1) * blockSize + 1,
                        j * blockSize + 1, (j + 1) * blockSize + 1);
                requests.clear();
                if (i != 0)
                {
                    // send old top row
                    requests.push_back(
                        world.isend(world.rank() + 1, MakeMessageTag("orow"),
                            ublas::vector<double>{ublas::row(currentBlockRow, 1)}));
                    // receive top row
                    requests.push_back(
                        world.irecv(world.rank() + 1, MakeMessageTag("nrow"), tempVector1));
                }
                if (j != gridSize - 1)
                {
                    // receive old right column
                    requests.push_back(
                        world.irecv(world.rank() + 1, MakeMessageTag("ocol"), tempVector2));
                }
                mpi::wait_all(requests.begin(), requests.end());
                if (i != 0)
                {
                    ublas::row(currentBlockRow, 0) = tempVector1;
                }
                auto currentBlockColumn = 
                    ublas::subrange(matrix,
                        i * blockSize + 1, (i + 1) * blockSize + 1,
                        j * blockSize + 1, (j + 1) * blockSize + 2);
                if (j != gridSize - 1)
                {
                    ublas::column(currentBlockColumn, currentBlockColumn.size2() - 1) = tempVector2;
                }
                iterationDelta = GaussSeidel(ublas::subrange(matrix,
                    i * blockSize, (i + 1) * blockSize + 2,
                    j * blockSize, (j + 1) * blockSize + 2), h);
                if (iterationDelta > procDelta)
                {
                    procDelta = iterationDelta;
                }
                if (j != gridSize - 1)
                {
                    // send right column
                    mpi::request request =
                        world.isend(world.rank() + 1, MakeMessageTag("ncol"),
                            ublas::vector<double>{ublas::column(currentBlockColumn, blockSize - 2)});
                    request.wait();
                }
                ++i;
            }
            // column
            for (; i < gridSize; ++i)
            {
                auto currentBlock = ublas::subrange(matrix,
                        i * blockSize + 1, (i + 1) * blockSize + 1,
                        j * blockSize, (j + 1) * blockSize + 2);
                requests.clear();
                // send old left column
                requests.push_back(
                    world.isend(world.rank() - 1, MakeMessageTag("ocol"),
                        ublas::vector<double>{ublas::column(currentBlock, 1)}));
                // receive left column
                requests.push_back(
                    world.irecv(world.rank() - 1, MakeMessageTag("ncol"), tempVector1));
                if (j != gridSize - 1)
                {
                    // receive old right column
                    requests.push_back(
                        world.irecv(world.rank() + 1, MakeMessageTag("ocol"), tempVector2));
                }
                mpi::wait_all(requests.begin(), requests.end());
                ublas::column(currentBlock, 0) = tempVector1;
                if (j != gridSize - 1)
                {
                    ublas::column(currentBlock, currentBlock.size2() - 1) = tempVector2;
                }
                iterationDelta = GaussSeidel(ublas::subrange(matrix,
                    i * blockSize, (i + 1) * blockSize + 2,
                    j * blockSize, (j + 1) * blockSize + 2), h);
                
                if (iterationDelta > procDelta)
                {
                    procDelta = iterationDelta;
                }
                if (j != gridSize - 1)
                {
                    // send right column
                    mpi::request request =
                        world.isend(world.rank() + 1, MakeMessageTag("ncol"),
                            ublas::vector<double>{ublas::column(currentBlock, currentBlock.size1() - 2)});
                    request.wait();
                }
                
            }
        }
        mpi::all_reduce(world, procDelta, delta, mpi::maximum<double>());
    } while (delta > eps);
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - currentTime).count() << std::endl;
    if (world.rank() != 0)
    {
        mpi::gather(world, matrix, 0);
    }
    else
    {
        std::vector<ublas::matrix<double>> results;
        mpi::gather(world, matrix, results, 0);
        auto resultSubmatrix = ublas::subrange(matrix, 1, n - 1, 1, n - 1);
        auto index = 0;
        for (auto iter = results.rbegin(); iter != --results.rend(); ++iter, ++index)
        {
            auto result = ublas::subrange(*iter, 1, n - 1, 1, n - 1);
            auto i = index, j = 0;
            for (; j < gridSize - index; ++j)
            {
                ublas::subrange(resultSubmatrix,
                    i * blockSize, (i + 1) * blockSize,
                    j * blockSize, (j + 1) * blockSize) =
                    ublas::subrange(result,
                        i * blockSize, (i + 1) * blockSize,
                        j * blockSize, (j + 1) * blockSize);
            }
            j = gridSize - index - 1;
            ++i;
            for (; i < gridSize; ++i)
            {
                ublas::subrange(resultSubmatrix,
                    i * blockSize, (i + 1) * blockSize,
                    j * blockSize, (j + 1) * blockSize) =
                    ublas::subrange(result,
                        i * blockSize, (i + 1) * blockSize,
                        j * blockSize, (j + 1) * blockSize);
            }
        }
        Print(matrix);
    }
    return 0;
}

double f([[maybe_unused]] size_t i, [[maybe_unused]] size_t j, [[maybe_unused]] size_t n)
{
    return 0;
}

double u(size_t i, size_t j, size_t n)
{
    if (i == 0)
    {
        return (1 - 2 * static_cast<double>(j) / (n - 1)) * 100;
    }
    if (j == 0)
    {
        return (1 - 2 * static_cast<double>(i) / (n - 1)) * 100;
    }
    if (i == n - 1)
    {
        return (2 * static_cast<double>(j) / (n - 1) - 1) * 100;
    }
    if (j == n - 1)
    {
        return (2 * static_cast<double>(i) / (n - 1) - 1) * 100;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    const std::uniform_real_distribution<> dis(
        -100,
        std::nextafter(100, std::numeric_limits<double>::max()));
    return dis(gen);
}

ublas::vector<double> ReceiveVector(const mpi::communicator& communicator, const int source, const size_t size, const char* tag)
{
    ublas::vector<double> vector{ size };
    communicator.recv(source, MakeMessageTag(tag), vector);
    return vector;
}