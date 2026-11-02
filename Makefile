CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

main: main.cpp src/engine.cpp src/graph.cpp headers/engine.hpp headers/graph.hpp
	$(CXX) $(CXXFLAGS) main.cpp src/engine.cpp src/graph.cpp -o main

run: main
	./main

graph: main
	dot -Tsvg out/graph3.dot -o out/graph3.svg

.PHONY: run graph