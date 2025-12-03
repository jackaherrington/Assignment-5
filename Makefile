CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra
# OpenMP flags for macOS with Homebrew libomp (keg-only)
OMP_INCLUDE = /opt/homebrew/opt/libomp/include
OMP_LIB = /opt/homebrew/opt/libomp/lib
OMP_FLAGS = -Xpreprocessor -fopenmp -I$(OMP_INCLUDE)
LDFLAGS = -L$(OMP_LIB) -lomp

TARGET = monte_carlo_omp
SRC = monte_carlo_omp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

# Example runs (adjust N and schedule at runtime):
# OMP_NUM_THREADS=8 OMP_SCHEDULE=guided,1024 ./monte_carlo_omp x 10000000
# OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,1024 ./monte_carlo_omp cos100x 5000000
# OMP_NUM_THREADS=8 OMP_SCHEDULE=static,1024 ./monte_carlo_omp inv_sqrt 10000000
