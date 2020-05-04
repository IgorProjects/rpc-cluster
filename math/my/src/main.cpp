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

namespace mpi = boost::mpi;
namespace ublas = boost::numeric::ublas;

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

double GaussSeidel(ublas::matrix_range<ublas::matrix<double>> matrix, double h)
{
    const auto maxWaveSize = std::min(matrix.size1(), matrix.size2()) - 2;
    std::vector<double> deltas(maxWaveSize);
    for (auto waveSize = 1; waveSize < maxWaveSize + 1; ++waveSize) 
    {
#pragma omp parallel for shared(matrix,waveSize,deltas)
        for (int i = 1; i < waveSize + 1; ++i) 
        {
            auto j = waveSize + 1 - i;
            auto temp = matrix(i, j);
            matrix(i, j) = (matrix(i - 1, j) + matrix(i + 1, j) +
                matrix(i, j - 1) + matrix(i, j + 1) - h * h * f(i, j, maxWaveSize)) / 4;
            auto delta = fabs(temp - matrix(i, j));
            if (deltas[i - 1] < delta)
            {
                deltas[i - 1] = delta;
            }
        }
    }
//    for (auto wave = 0; wave < staticWaveNumbers; ++wave)
//    {
//#pragma omp parallel for shared(matrix,maxWaveSize,deltas)
//        for (int i = 1; i < maxWaveSize + 1; ++i)
//        {
//            auto j = maxWaveSize + wave + 2 - i;
//            auto temp = matrix(i, j);
//            matrix(i, j) = (matrix(i - 1, j) + matrix(i + 1, j) +
//                matrix(i, j - 1) + matrix(i, j + 1) - h * h * f(i, j, maxWaveSize)) / 4;
//            auto delta = fabs(temp - matrix(i, j));
//            if (deltas[i - 1] < delta)
//            {
//                deltas[i - 1] = delta;
//            }
//        }
//    }
    for (int waveSize = maxWaveSize - 1; waveSize > 0; --waveSize) 
    {
#pragma omp parallel for shared(matrix,waveSize,deltas)
        for (int i = maxWaveSize - waveSize + 1; i < maxWaveSize + 1; i++) {
            auto j = 2 * maxWaveSize - waveSize - i + 1;
            auto temp = matrix(i, j);
            matrix(i, j) = (matrix(i - 1, j) + matrix(i + 1, j) +
                matrix(i, j - 1) + matrix(i, j + 1) - h * h * f(i, j, maxWaveSize)) / 4;
            auto delta = fabs(temp - matrix(i, j));
            if (deltas[i - 1] < delta)
            {
                deltas[i - 1] = delta;
            }
        }
    }
    return *std::max_element(deltas.begin(), deltas.end());
}

auto GetMatrixBlock(ublas::matrix<double>& matrix, size_t i, size_t j, size_t blockSize)
{
    return ublas::subrange(matrix,
        i * blockSize, (i + 1) * blockSize + 2,
        j * blockSize, (j + 1) * blockSize + 2);
}

ublas::vector<double> ReceiveVector(const mpi::communicator& communicator, const int source, const size_t size, const char* tag);

int main(int argc, char* argv[])
{
#if 0
    constexpr auto n = 250;
    ublas::matrix<double> _matrix{n, n};
    auto matrix = ublas::subrange(_matrix, 0, n, 0 ,n);
    for (auto i = 0; i < n; ++i)
    {
        for (auto j = 0; j < n; ++j)
        {
            matrix(i, j) = u(i, j, n);
        }
    }
    //Print(matrix);
    double delta = 0;
    auto currentTime = std::chrono::high_resolution_clock::now();
    do
    {
        delta = GaussSeidel(matrix, 1 / static_cast<double>(n - 1));
    } while (delta > 0.1);
    std::cout << (std::chrono::high_resolution_clock::now() - currentTime).count() << std::endl;
    
    //Print(matrix);
#else
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
        std::cout << "Initial:\n";
        Print(matrix);
    }
    mpi::broadcast(world, matrix, 0);
    double procDelta = 0, delta = 0;
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
                    requests.push_back(
                        world.isend(world.rank() + 1, MakeMessageTag("orow"), 
                            ublas::vector<double>{ublas::row(currentBlock, 1)}));
                    requests.push_back(
                        world.irecv(world.rank() + 1, MakeMessageTag("nrow"), tempVector1));
                    //// send old top row
                    //world.send(world.rank() + 1, MakeMessageTag("orow"), ublas::vector<double>{ublas::row(matrix, 0)});
                    //// receive top row
                    //ublas::row(matrix, 0) = ReceiveVector(world, world.rank() + 1, currentBlock.size2(), "nrow");
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
                    mpi::request request =
                        world.isend(world.rank() - 1, MakeMessageTag("nrow"),
                            ublas::vector<double>{ublas::row(currentBlock, currentBlock.size1() - 2)});
                    request.wait();
                    //// send bottom row
                    //SendRow(world, matrix, i, j, blockSize);
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
                    requests.push_back(
                        world.isend(world.rank() + 1, MakeMessageTag("orow"),
                            ublas::vector<double>{ublas::row(currentBlockRow, 1)}));
                    requests.push_back(
                        world.irecv(world.rank() + 1, MakeMessageTag("nrow"), tempVector1));
                    // send old top row

                    // receive top row
                    //ReceiveRow(world, matrix, i, j, blockSize);
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
                    mpi::request request =
                        world.isend(world.rank() + 1, MakeMessageTag("ncol"),
                            ublas::vector<double>{ublas::column(currentBlockColumn, blockSize - 2)});
                    request.wait();
                    //// send right column
                    //SendColumn(world, matrix, i, j, blockSize, lastBlockSize);
                }
                ++i;
            }
            // column
            // currentBlock = GetMatrixBlock(matrix, i, j, gridSize, blockSize, lastBlockSize);
            for (; i < gridSize; ++i)
            {
                auto currentBlock = ublas::subrange(matrix,
                        i * blockSize + 1, (i + 1) * blockSize + 1,
                        j * blockSize, (j + 1) * blockSize + 2);
                requests.clear();
                requests.push_back(
                    world.isend(world.rank() - 1, MakeMessageTag("ocol"),
                        ublas::vector<double>{ublas::column(currentBlock, 1)}));
                requests.push_back(
                    world.irecv(world.rank() - 1, MakeMessageTag("ncol"), tempVector1));
                //// send old left column
                //// receive left column
                //ReceiveColumn(world, matrix, i, j, blockSize, lastBlockSize);
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
                    mpi::request request =
                        world.isend(world.rank() + 1, MakeMessageTag("ncol"),
                            ublas::vector<double>{ublas::column(currentBlock, currentBlock.size1() - 2)});
                    request.wait();
                    // send right column
                    // SendColumn(world, matrix, i, j, blockSize, lastBlockSize);
                }
                
            }
        }
        mpi::all_reduce(world, procDelta, delta, mpi::maximum<double>());
    } while (delta > eps);

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
        std::cout << "Result:\n";
        Print(matrix);
    }
#endif

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