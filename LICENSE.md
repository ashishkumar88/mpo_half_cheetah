## Maze Solver

This project contains a C++ program developed to solve maze tour using classical AI techniques

### Approach

The technical challenge present various user stories that need to be solved. The approaches applied to solve the user stories are mentioned below.

- User Story 1 - The requirement is to find an empty space in a single row. This is performed by performing a linear search on a row.
    - Assumptions - The Maze is either a 1D or a 2D maze. The program is expected to return the *first* empty space in a row if multiple empty spaces exist.
- User Story 2 - The requirement is to walk through a hallway in a hallway maze. The first scenario is when a column is a hallway. The walk is performed by finding a cell from the first row that is empty and marking it as a start cell. Futhermore, walk through that column until a wall is hit. The second scenario is when a row is a hallway. The walk is performed by finding a cell from the first column that is empty and marking it as a start cell. Futhermore, walk through that row until a wall is hit.
    - Assumptions - The Maze is either a 1D or a 2D maze. A *hallway* maze that contains either a single row hallway with empty spaces or a single column hallway with empty spaces. Provided maze contains only one hallway. The program does not work on multiple hallways.
- User Story 3 - The requirement is to find a way into a rectangular room and a way out of it. The program find a path into this rectangular room and out of it by reusing the methods implemented for the previous use story. 
    - Assumptions - The Maze is a 2D maze. A *room* is a rectangular composition of cells which are empty and is atleast two cells wide. A room always has an empty cell or door outside of it and adjacent to the top left corner and another empty cell or door adjacent to a bottom right corner. The program assumses that maze has **only one** room. The program does not provide paths for multiple rooms.
- User Story 4 - The requirement is to find a winding path if it exists in a maze. The program find a path into a winding path and out of it. This is solved by using a graph algorithm called depth first search or DFS. The program finds the start of the winding path and then follows it.
    - Assumptions - The Maze is a 2D maze. A *winding path* does not contain forks, that is, a cell never creates two subpaths. The program assumses that maze has **only one** winding path and each path is exactly **one** cell wide. The program does not provide paths for multiple winding paths.
- User Story 5 - The requirement is to find a winding path if it exists in a maze. The program find a path into a winding path and out of it. This is solved by using a graph algorithm called depth first search or DFS. The program finds the start of the winding path and then follows it.
    - Assumptions - The Maze is a 2D maze. A *winding path* does not contain forks, that is, a cell never creates two subpaths. The program assumses that maze has **only one** winding path and each path is exactly **one** cell wide. The program does not provide paths for multiple winding paths. Only forward, backward, left and right movements are allowed.

### Project Structure

The project has the following structure:
- apps - This directory contains the `solver.cpp` which contains the main method. 
- include - This directory contains the `maze.hpp` header file which contains the class and function declaractions.
- src - This directory contains various .cpp files which contain function and class definitions. 
- tests - This directory contains the unit and integration tests.

### How Tos

#### Build

To build the project, the easiest approach is to execute the [`build.sh`](scripts/build.sh) script in the `scripts` directory when the current directory is the root directory. This script creates a directory named `build` under the root directory and creates all the build related files in this directory. The executable is created under the `build/bin` subdirectory. The `build.sh` can also be executed when the current directory is any other directory in the filesystem. Similar to above, the scripts creates a `build` directory. This build has only been tested on an Ubuntu 20.04 based computer. The Google Tests are executed after build is done and no additional command has to be executed.

To build, run the following commands:
```bash
cd directory/under/which/build/shall/be/performed
path/to/build.sh
```

If you are not sure that the sytem contains all the required dependencies for building this program, then run the build script with the `--prereq` flag as shown below.
```bash
cd directory/under/which/build/shall/be/performed
path/to/build.sh --prereq
```

To clean a previous build, use the `--clean` flag when running the build script.
```bash
cd directory/under/which/build/shall/be/performed
path/to/build.sh --clean
```

Expected output is below:

<details>
  <summary>Click to expand</summary>

    ```
    Clearning the previous builds.
    Building the project
    -- The CXX compiler identification is GNU 9.4.0
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- The C compiler identification is GNU 9.4.0
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Found Python3: /usr/bin/python3.9 (found version "3.9.18") found components: Interpreter
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
    -- Looking for pthread_create in pthreads
    -- Looking for pthread_create in pthreads - not found
    -- Looking for pthread_create in pthread
    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /home/ashish/build
    Scanning dependencies of target MazeSolverLib
    [  6%] Building CXX object src/CMakeFiles/MazeSolverLib.dir/grid.cpp.o
    [ 12%] Building CXX object src/CMakeFiles/MazeSolverLib.dir/graph.cpp.o
    [ 18%] Building CXX object src/CMakeFiles/MazeSolverLib.dir/utils.cpp.o
    [ 25%] Linking CXX static library libMazeSolverLib.a
    [ 25%] Built target MazeSolverLib
    Scanning dependencies of target MazeSolver
    [ 31%] Building CXX object apps/CMakeFiles/MazeSolver.dir/solver.cpp.o
    [ 37%] Linking CXX executable ../bin/MazeSolver
    [ 37%] Built target MazeSolver
    Scanning dependencies of target gtest
    [ 43%] Building CXX object _deps/googletest-build/googletest/CMakeFiles/gtest.dir/src/gtest-all.cc.o
    [ 50%] Linking CXX static library ../../../lib/libgtest.a
    [ 50%] Built target gtest
    Scanning dependencies of target gmock
    [ 56%] Building CXX object _deps/googletest-build/googlemock/CMakeFiles/gmock.dir/src/gmock-all.cc.o
    [ 62%] Linking CXX static library ../../../lib/libgmock.a
    [ 62%] Built target gmock
    Scanning dependencies of target gmock_main
    [ 68%] Building CXX object _deps/googletest-build/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
    [ 75%] Linking CXX static library ../../../lib/libgmock_main.a
    [ 75%] Built target gmock_main
    Scanning dependencies of target gtest_main
    [ 81%] Building CXX object _deps/googletest-build/googletest/CMakeFiles/gtest_main.dir/src/gtest_main.cc.o
    [ 87%] Linking CXX static library ../../../lib/libgtest_main.a
    [ 87%] Built target gtest_main
    Scanning dependencies of target test_grid
    [ 93%] Building CXX object tests/CMakeFiles/test_grid.dir/test_grid.cpp.o
    [100%] Linking CXX executable ../bin/test_grid
    Run grid tests
    [==========] Running 16 tests from 1 test suite.
    [----------] Global test environment set-up.
    [----------] 16 tests from GridTest
    [ RUN      ] GridTest.Constructor
    [       OK ] GridTest.Constructor (0 ms)
    [ RUN      ] GridTest.Constructor2
    File does not exist : "maps/does_not_exist.txt"
    filesystem error: File does not exist.
    [       OK ] GridTest.Constructor2 (0 ms)
    [ RUN      ] GridTest.Constructor3
    File does not exist : ""
    filesystem error: File does not exist.
    [       OK ] GridTest.Constructor3 (0 ms)
    [ RUN      ] GridTest.Constructor4
    [       OK ] GridTest.Constructor4 (0 ms)
    [ RUN      ] GridTest.Constructor5
    [       OK ] GridTest.Constructor5 (0 ms)
    [ RUN      ] GridTest.SearchARowForEmptySpace1
    [       OK ] GridTest.SearchARowForEmptySpace1 (0 ms)
    [ RUN      ] GridTest.SearchARowForEmptySpace2
    [       OK ] GridTest.SearchARowForEmptySpace2 (0 ms)
    [ RUN      ] GridTest.SearchARowForEmptySpace3
    Row index is out of bounds.
    [       OK ] GridTest.SearchARowForEmptySpace3 (0 ms)
    [ RUN      ] GridTest.SearchARowForEmptySpace4
    [       OK ] GridTest.SearchARowForEmptySpace4 (0 ms)
    [ RUN      ] GridTest.SearchARowForEmptySpace5
    Invalid map file : "maps/multiple_rows_2.txt"
    Invalid map file.
    [       OK ] GridTest.SearchARowForEmptySpace5 (0 ms)
    [ RUN      ] GridTest.HallwayWalk1
    [       OK ] GridTest.HallwayWalk1 (0 ms)
    [ RUN      ] GridTest.HallwayWalk2
    [       OK ] GridTest.HallwayWalk2 (0 ms)
    [ RUN      ] GridTest.HallwayWalk3
    [       OK ] GridTest.HallwayWalk3 (0 ms)
    [ RUN      ] GridTest.HallwayWalk4
    [       OK ] GridTest.HallwayWalk4 (0 ms)
    [ RUN      ] GridTest.HallwayWalk5
    [       OK ] GridTest.HallwayWalk5 (0 ms)
    [ RUN      ] GridTest.HallwayWalk6
    [       OK ] GridTest.HallwayWalk6 (0 ms)
    [----------] 16 tests from GridTest (0 ms total)

    [----------] Global test environment tear-down
    [==========] 16 tests from 1 test suite ran. (0 ms total)
    [  PASSED  ] 16 tests.
    [100%] Built target test_grid
    /home/ashish
    Build complete
    ```
</details>


#### Execution

To run the program, prepare a file that encodes a maze has to be created. Creating a text file that encodes a maze using 1s and 0s involves the below steps.
- Design maze layout - First create a layout of a MxN grid, where M is the number of rows and N is the number of columns. Plan the pathways and walls. Pathways must be represented by '0's (indicating free space) and walls by '1's (indicating blocked space).
- Text Editor - Open a text editor on your computer. Avoid using word processors, as they add additional formatting that can interfere with the maze structure.
- Encode the Maze: Start encoding the maze into the text file. Begin from the top-left corner of your maze design, translating each cell into a '1' for a wall or a '0' for a path. Enter the corresponding number for each cell in your maze grid, moving from left to right. Please **do not** use spaces or any other characters.
- Use Line Breaks for Rows: At the end of each row of the maze, press 'Enter' to start a new line. This will ensure that each row of your maze is on a separate line in the text file, maintaining the grid structure. Ensure that the number of columns are consistent. This program only accepts a MxN grid.
- Save the File: Once you've finished encoding the entire maze, save the file. Choose a suitable file name and ensure it has a '.txt' extension. For example, you might name it `maze.txt`.

After the maze has been created, run the solver using the following commands:
```bash
cd to/directory/under/which/build/was/performed
./build/bin/MazeSolver -m path/to/maze/file <options>
```

The `MazeSolver` executable requires path to the maze file. Optionally, a user story id can also be provided to the program. If user story id is not provided, the program will default to the highest user story implement in the last commit. Expected output for different user stories are shown below:


- User Story 1
```
./build/bin/MazeSolver -m path/to/maze/file -u 1
First empty space in row 1 is at column 2
```

- User Story 2
```
./build/bin/MazeSolver -m path/to/maze/file -u 2
Path Start -> (0, 3) -> (1, 3) -> (2, 3) -> (3, 3) -> (4, 3) -> (5, 3) -> End
```

- User Story 3
```
./build/bin/MazeSolver -m path/to/maze/file -u 3
Path Start -> (0, 1) -> (1, 1) -> (2, 1) -> (2, 2) -> (2, 3) -> (2, 4) -> (3, 4) -> End
```

- User Story 4
```
./build/bin/MazeSolver -m path/to/maze/file -u 4
Path Start -> (0, 1) -> (1, 1) -> (1, 2) -> (1, 3) -> (2, 3) -> (3, 3) -> (3, 2) -> (3, 1) -> (4, 1) -> (5, 1) -> (5, 2) -> (5, 3) -> (6, 3) -> End
```

- User Story 5
```
./build/bin/MazeSolver -m path/to/maze/file # or -u 5
Path Start -> (0, 1) -> (1, 1) -> (1, 2) -> (1, 3) -> (2, 3) -> (3, 3) -> (3, 2) -> (3, 1) -> (4, 1) -> (5, 1) -> (5, 2) -> (5, 3) -> (6, 3) -> End
```
### Reflections/Analysis

#### Analysis Story 1
Although the maze solver can solve various scenarios, it also makes assumptions which are specified in the first section. The maze solver will not provide multiple paths in case multiple start and/or multiple ends exit. The solver has not been stress tested and may run into performance issues when large mazes are provided. The solution also assumes that only forward, backward, left and right movements are allowed. As diagonal movements are not allowed, the solution will fail in mazes that requires diagonal movements.

#### Analysis Story 2
I believe the requirement was to come up with a simple solution, however, I implement A star for the User Story 5 as the amount of effort coding effort required to continue with the simple effort and implement a solution seemed to be higher than implementing A star. Now, A star is optimal and complete if the heuristic is admissible and monotonic. Now, the manhattan distance is an admissible heuristic as the manhattan distance is never higher than the lowest possible cost in grids in which forward, backward, left and right movements are allowed. Additionally, the manhattan distance is increasingly monotonic, that is it either increases or stays the same. Hence, the solution is optimal. The worst case time complexity of A star in this case could be that of Djikstra's which is O((M\*N) log(M*N)).

On the programming side, some future work:
1. Explore a way to use priority queue
2. Find a way to avoid using an additional map to store child to parent relationship
3. Find a way to avoid the O(n) search starting at line 287 of graph.cpp

#### Analysis Story 3
To decompose the problem of navigating a 1x3 "ship" through a maze with the ability to move forward, backward, and rotate around its center of gravity, we can follow an incremental approach similar to that used for simpler maze navigation problems. Here are the steps to break down the problem:

1. User Story 1 -
Implement the ship's ability to move forward and backward in a row or column hallway in a grid.

2. User Story 2 -
Enable the ship to rotate around its center of gravity. Ensure that the rotation is accurate and consistent. 

3. User Story 3 -
Implement the ship entering a room which has enough space to rotate and exit. Implement basic collision detection for the ship. This includes preventing the ship from moving through the maze walls. An example maze could look like below:
```
11101111
11101111
11101111
10000001
10000001
10000001
11111011
11111011
```

4. User Story 4 - Enhance the ship's movement to allow for more precise control, such as slight adjustments in position. Implement more complex rotation mechanics, allowing the ship to navigate around corners and tight spaces. An exmple maze could look like below:
```
11101111
11101111
11101111
10000001
10000001
10000001
11111011
11111011
11111011
11111011
10000001
10000001
10000001
11101111
11101111
11101111
```

Implementation should treat the following map as invalid:
```
11101111
11101111
11101111
10000001
10000001
10000001
11111011
11111011
11111011
11111011
11100011
11101111
11101111
11101111
```

5. User Story 5 - Introduce more complex mazes with varied layouts, including tight turns and narrow passages as provided in the problem statement.

##### Solution 
To solve a maze for a 1xR robot, a collision model shall be developed. A star can still be used with a neighor validation check. Only valid neighbor should be considered for adding to the open list. A neighbor is valid if the 1xR robot can move into it, where a neighbor is an adjacent cell with respect to the center of gravity of the robot. Consider the following map:

```
11101111
111R1111
111R1111
100R0001
10000001
10000001
```
In the above example, the center of the gravity is (3, 4), using 1-indexing. (2, 4) and (4, 4) are the only valid neighbors.

Let's consider another example:
```
11101111
11101111
11101111
100R0001
000R0001
100R0001
11101111
11101111
```
In the above example, the center of the gravity is (5, 4), using 1-indexing. There are 4 valid neighbors: (4, 4), (6, 4), (5, 3) and (5, 5).

Along with a path, the sequence of actions must also be provided.

### License
This program is licensed under GPL 3, license [here](LICENSE.md).
