CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

main: main.cpp src/engine.cpp headers/engine.hpp
	$(CXX) $(CXXFLAGS) main.cpp src/engine.cpp -o main

run: main
	./main

clean:
	rm -f main

.PHONY: run clean