.PHONY: clean
.DEFAULT_GOAL: main.exe

CC := g++
CCMPI := mpicxx
CFLAGS := -Wall -O3 -pedantic -std=c++11 #-lstdc++fs
ARMAFLAGS := -larmadillo -framework Accelerate 
OBJECTS := main.o
VPATH := src
CLASSES := ansatz.cpp model.cpp sampler.cpp read_options.cpp random.cpp
HEADERS := library.h

main.exe : $(OBJECTS)
	$(CCMPI) $(OBJECTS) -o main.exe $(CFLAGS) $(ARMAFLAGS)
main.o : main.cpp $(CLASSES) $(HEADERS)
	$(CCMPI) -c main.cpp -o main.o $(CFLAGS)

clean :
	-rm *.o *.dat *.wf main.exe
	-rm -rf "./CONFIG"

clean_file:
	-rm *.dat *.wf
	-rm -rf "./CONFIG"

clean_ingredients:
	-rm *.o main.exe
