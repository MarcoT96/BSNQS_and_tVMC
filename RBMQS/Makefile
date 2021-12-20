.PHONY: clean
.DEFAULT_GOAL: main.exe

CC = g++
CFLAGS = -Wall -std=c++11 -O3
OBJECTS = main.o
VPATH = src
CLASSES = RBM.cpp models.cpp sampler.cpp readOptions.cpp random.cpp
HEADERS = library.h

main.exe : $(OBJECTS)
	$(CC) $(OBJECTS) -o main.exe $(CFLAGS) -framework Accelerate
main.o : main.cpp $(CLASSES) $(HEADERS)
	$(CC) -c main.cpp -o main.o $(CFLAGS)
clean :
	rm *.o main.exe
