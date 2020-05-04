#pragma warning(disable:4996)
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>

#include <random>

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

double f(double x, double y);
double u(double x, double y);

double GaussSeidel(ublas::matrix_range<ublas::matrix<double>> matrix)
{
    matrix(1 ,1) = 999;

    return 0.01;
}

auto GetMatrixBlock(ublas::matrix<double>& matrix, size_t i, size_t j, size_t gridSize, size_t blockSize, size_t lastBlockSize)
{
    std::cout << "Get (" << i << ", " << j << ") block" << std::endl;
    return ublas::subrange(matrix,
        i * blockSize, i * blockSize + (i == gridSize - 1 ? lastBlockSize : blockSize) + 2,
        j * blockSize, j * blockSize + (j == gridSize - 1 ? lastBlockSize : blockSize) + 2);
}

ublas::vector<double> ReceiveVector(const mpi::communicator& communicator, const int source, const size_t size, const char* tag);
//void SendRow(
//    const mpi::communicator& communicator, 
//    ublas::matrix<double>& matrix,
//    size_t i,
//    size_t j,
//    size_t size);
//void ReceiveColumn(
//    const mpi::communicator& communicator, 
//    ublas::matrix<double>& matrix,
//    size_t i,
//    size_t j,
//    size_t blockSize,
//    size_t columnSize);
//void SendColumn(
//    const mpi::communicator& communicator, 
//    ublas::matrix<double>& matrix,
//    size_t i,
//    size_t j,
//    size_t blockSize,
//    size_t columnSize);


int main(int argc, char* argv[])
{
    mpi::environment env;
    mpi::communicator world;
    const auto gridSize = world.size() - 1;

    ublas::matrix<double> matrix;
    size_t n;
    double eps;
    size_t blockSize, lastBlockSize;
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
        blockSize = (n - 2) / gridSize;
        lastBlockSize = blockSize + (n - 2) % gridSize;
    }
    mpi::broadcast(world, blockSize, 0);
    mpi::broadcast(world, lastBlockSize, 0);
    mpi::broadcast(world, eps, 0);
    mpi::broadcast(world, n, 0);
    matrix.resize(n, n);
    if (world.rank() == 0)
    {
        for (auto i = 0; i < n; ++i)
        {
            for (auto j = 0; j < n; ++j)
            {
                matrix(i, j) = u(i, j);
            }
        }
        std::cout << "Initial:" << matrix << std::endl;
    }
    mpi::broadcast(world, matrix, 0);
    double procDelta = 0, delta = 0;
    do
    {
        if (world.rank() != 0)
        {
            auto j = 0, i = gridSize - world.rank();
            double iterationDelta;
            auto currentBlock = GetMatrixBlock(matrix, i, j, gridSize, blockSize, lastBlockSize);
            // row
            for (; j < world.rank() - 1; ++j, currentBlock = GetMatrixBlock(matrix, i, j, gridSize, blockSize, lastBlockSize))
            {
                if (i != 0)
                {
                    std::cout << "Row: Send/Receive begin " << i << ' ' << j << std::endl;
                    ublas::vector<double> tempVector{currentBlock.size2()};
                    world.sendrecv(
                        world.rank() + 1, MakeMessageTag("orow"), ublas::vector<double>{ublas::row(currentBlock, 1)},
                        world.rank() + 1, MakeMessageTag("nrow"), tempVector);
                    ublas::row(currentBlock, 0) = tempVector;
                    std::cout << "Row: Send/Receive end " << i << ' ' << j << std::endl;
                    //// send old top row
                    //world.send(world.rank() + 1, MakeMessageTag("orow"), ublas::vector<double>{ublas::row(matrix, 0)});
                    //// receive top row
                    //ublas::row(matrix, 0) = ReceiveVector(world, world.rank() + 1, currentBlock.size2(), "nrow"); // TODO: rank!
                }
                if (i != gridSize - 1)
                {
                    std::cout << "Row: Receive old begin " << i << ' ' << j << std::endl;
                    // receive old bottom row
                    ublas::row(currentBlock, currentBlock.size1() - 1) = 
                        ReceiveVector(world, world.rank() - 1, currentBlock.size2(), "orow");
                    std::cout << "Row: Receive old end " << i << ' ' << j << std::endl;
                }
                iterationDelta = GaussSeidel(currentBlock);
                if (iterationDelta > procDelta)
                {
                    procDelta = iterationDelta;
                }
                if (i != gridSize - 1)
                {
                    std::cout << "Row: Send begin " << i << ' ' << j << std::endl;
                    world.send(world.rank() - 1, MakeMessageTag("nrow"), 
                        ublas::vector<double>{ublas::row(currentBlock, currentBlock.size1() - 2)});
                    std::cout << "Row: Send end " << i << ' ' << j << std::endl;
                    //// send bottom row
                    //SendRow(world, matrix, i, j, blockSize);
                }
            }
            // angle
            if (i != 0)
            {
                std::cout << "Angle: Send/Receive begin " << i << ' ' << j << std::endl;
                ublas::vector<double> tempVector{ currentBlock.size2() };
                world.sendrecv(
                    world.rank() + 1, MakeMessageTag("orow"), ublas::vector<double>{ublas::row(currentBlock, 1)},
                    world.rank() + 1, MakeMessageTag("nrow"), tempVector);
                ublas::row(currentBlock, 0) = tempVector;
                std::cout << "Angle: Send/Receive end " << i << ' ' << j << std::endl;
                // send old top row

                // receive top row
                //ReceiveRow(world, matrix, i, j, blockSize);
            }
            if (j != gridSize - 1)
            {
                std::cout << "Angle: Receive old begin " << i << ' ' << j << std::endl;
                // receive old right column
                ublas::column(currentBlock, currentBlock.size2() - 1) = 
                        ReceiveVector(world, world.rank() - 1, currentBlock.size1(), "ocol");
                std::cout << "Angle: Receive old end " << i << ' ' << j << std::endl;
            }
            iterationDelta = GaussSeidel(currentBlock);
            if (iterationDelta > procDelta)
            {
                procDelta = iterationDelta;
            }
            if (j != gridSize - 1)
            {
                std::cout << "Angle: Send begin " << i << ' ' << j << std::endl;
                world.send(world.rank() + 1, MakeMessageTag("ncol"), 
                    ublas::vector<double>{ublas::column(currentBlock, currentBlock.size1() - 2)});
                std::cout << "Angle: Send end " << i << ' ' << j << std::endl;
                //// send right column
                //SendColumn(world, matrix, i, j, blockSize, lastBlockSize);
            }
            ++i;
            // column
            currentBlock = GetMatrixBlock(matrix, i, j, gridSize, blockSize, lastBlockSize);
            for (; i < gridSize; ++i, currentBlock = GetMatrixBlock(matrix, i, j, gridSize, blockSize, lastBlockSize))
            {
                if (j != 0)
                {
                    std::cout << "Column: Send/Receive begin " << i << ' ' << j << std::endl;
                    ublas::vector<double> tempVector{ currentBlock.size1() };
                    world.sendrecv(
                        world.rank() - 1, MakeMessageTag("ocol"), ublas::vector<double>{ublas::column(currentBlock, 1)},
                        world.rank() - 1, MakeMessageTag("ncol"), tempVector);
                    ublas::column(currentBlock, 0) = tempVector;
                    std::cout << "Column: Send/Receive end " << i << ' ' << j << std::endl;
                    //// send old left column
                    //// receive left column
                    //ReceiveColumn(world, matrix, i, j, blockSize, lastBlockSize);
                }
                if (j != gridSize - 1)
                {
                    std::cout << "Column: Receive old begin " << i << ' ' << j << std::endl;
                    // receive old right column
                    ublas::column(currentBlock, currentBlock.size2() - 1) = 
                        ReceiveVector(world, world.rank() - 1, currentBlock.size1(), "ocol");
                    std::cout << "Column: Receive old end " << i << ' ' << j << std::endl;
                }
                iterationDelta = GaussSeidel(currentBlock);
                if (iterationDelta > procDelta)
                {
                    procDelta = iterationDelta;
                }
                if (j != gridSize - 1)
                {
                    std::cout << "Column: Send begin " << i << ' ' << j << std::endl;
                    world.send(world.rank() + 1, MakeMessageTag("ncol"), 
                        ublas::vector<double>{ublas::column(currentBlock, currentBlock.size1() - 2)});
                    std::cout << "Column: Send end " << i << ' ' << j << std::endl;
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
        for (auto iter = results.rbegin(); iter != results.rend(); ++iter, ++index)
        {
            auto result = ublas::subrange(*iter, 1, n - 1, 1, n - 1);
            auto i = index, j = 0;
            for (; j < gridSize - index; ++j)
            {
                ublas::subrange(resultSubmatrix,
                    i* blockSize, i* blockSize + (i == gridSize - 1 ? lastBlockSize : blockSize),
                    j* blockSize, j* blockSize + (j == gridSize - 1 ? lastBlockSize : blockSize)) =
                    ublas::subrange(result,
                        i * blockSize, i * blockSize + (i == gridSize - 1 ? lastBlockSize : blockSize),
                        j * blockSize, j * blockSize + (j == gridSize - 1 ? lastBlockSize : blockSize));
                //GetMatrixBlock(matrix, i, j, gridSize, blockSize, lastBlockSize) = 
                //    GetMatrixBlock(*iter, i, j, gridSize, blockSize, lastBlockSize);
            }
            j = gridSize - index - 1;
            ++i;
            for (; i < gridSize; ++i)
            {
                ublas::subrange(resultSubmatrix,
                    i* blockSize, i* blockSize + (i == gridSize - 1 ? lastBlockSize : blockSize),
                    j* blockSize, j* blockSize + (j == gridSize - 1 ? lastBlockSize : blockSize)) =
                    ublas::subrange(result,
                        i * blockSize, i * blockSize + (i == gridSize - 1 ? lastBlockSize : blockSize),
                        j * blockSize, j * blockSize + (j == gridSize - 1 ? lastBlockSize : blockSize));
            }
        }
        std::cout << "Result:\n" << matrix << std::endl;
    }
    
    

    return 0;
}

double f([[maybe_unused]] double x, [[maybe_unused]] double y)
{
    return 0;
}

double u(double x, double y)
{
    if (y == 0)
    {
        return (1 - x - x) * 100;
    }
    if (x == 0)
    {
        return (1 - y - y) * 100;
    }
    if (y == 1)
    {
        return (x + x - 1) * 100;
    }
    if (x == 1)
    {
        return (y + y - 1) * 100;
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
//void SendRow(
//    const mpi::communicator& communicator, 
//    ublas::matrix<double>& matrix,
//    size_t i,
//    size_t j,
//    size_t size)
//{
//    const ublas::vector<double> vector = ublas::subrange(
//        ublas::row(matrix, (i + 1) * size - 1),
//        j * size, (j + 1) * size);
//    communicator.send(communicator.rank() - 1, MakeMessageTag("row"), vector);
//}
//void ReceiveColumn(
//    const mpi::communicator& communicator, 
//    ublas::matrix<double>& matrix,
//    size_t i,
//    size_t j,
//    size_t blockSize,
//    size_t columnSize)
//{
//    ublas::vector<double> vector{ columnSize };
//    communicator.recv(communicator.rank() - 1, MakeMessageTag("col"), vector);
//    std::copy(vector.begin(), vector.end(), 
//        ublas::column(matrix, j * blockSize - 1).begin() + i * blockSize);
//}
//void SendColumn(
//    const mpi::communicator& communicator, 
//    ublas::matrix<double>& matrix,
//    size_t i,
//    size_t j,
//    size_t blockSize,
//    size_t columnSize)
//{
//    const ublas::vector<double> vector = ublas::subrange(
//        ublas::column(matrix, (j + 1) * blockSize - 1),
//        i * blockSize, i * blockSize + columnSize);
//    communicator.send(communicator.rank() + 1, MakeMessageTag("col"), vector);
//}