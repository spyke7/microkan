CXX = g++
CXXFLAGS = -Wall -Wextra

main: main.cpp src/engine.cpp src/graph.cpp src/spline.cpp src/kan.cpp headers/engine.hpp headers/graph.hpp headers/spline.hpp headers/kan.hpp
	$(CXX) $(CXXFLAGS) main.cpp src/engine.cpp src/graph.cpp src/kan.cpp src/spline.cpp -o main

run: main
	./main

graph: main
	dot -Tsvg out/graph4.dot -o out/graph4.svg

.PHONY: run graph