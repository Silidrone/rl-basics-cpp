## TagGame Exercise Setup

For the TagGame exercise, you need to run TWO components in order:

**IMPORTANT: The Java game MUST be run FIRST, then the RL agent!**

### Quick Start (Using Shell Scripts)

For convenience, you can use the provided shell scripts that automatically handle dependency installation:

1. **Run the TagGame (in terminal 1):**
   ```bash
   ./run_taggame.sh
   ```
   Wait for "Server started on port 12345" before proceeding.

2. **Run the RL Agent (in terminal 2):**
   ```bash
   ./run_rl.sh
   ```

The scripts will automatically check for and install required dependencies (maven, cmake, nlohmann-json3-dev) if needed.

### Manual Setup (Alternative)

#### 1. Running the Java Game (Run First!)

The Java game is located in `./taggame-java` (code based on Dr. Dindar Öz's implementation, modified for socket communication).

##### Install dependencies

```bash
sudo apt install maven
```

##### Compile and run

```bash
cd taggame-java
mvn compile
java -cp "target/classes:lib/geom2D/javaGeom-0.11.1.jar:lib/slick2D/slick.jar:lib/slick2D/lwjgl.jar:lib/slick2D/lwjgl_util.jar:$HOME/.m2/repository/org/json/json/20231013/json-20231013.jar" -Djava.library.path=natives taggame.SlickGraphicsRunner
```

Wait for the Java game to start and display "Server started on port 12345" before proceeding.

#### 2. Running the RL Agent (Run Second!)

After the Java game is running, build and run the RL agent:

##### Install dependencies

```bash
sudo apt update && sudo apt install cmake nlohmann-json3-dev -y
```

##### Build and run

```bash
cmake -S . -B build && cmake --build build
./build/output_executable
```

The RL agent will connect to the Java game server at `127.0.0.1:12345` and begin training.

## Testing Different Algorithms and Environments

To test different algorithms or environments, you need to modify `main.cpp` and rebuild the project.

### Available Environments

1. **TagGame** (requires Java game running)
   - Function Approximation TD: `#include "taggame/fa_td_solution.h"` → `taggame_main()`
   - Tabular TD: `#include "taggame/td_solution.h"` → `taggame_main()`

2. **Windy Gridworld** (Exercise 6.9)
   - Function Approximation TD: `#include "barto_sutton_exercises/6_9/fa_td_solution.h"` → `windygridworld_main()`
   - Tabular TD: `#include "barto_sutton_exercises/6_9/td_solution.h"` → `windygridworld_main()`

3. **Blackjack** (Exercise 5.1)
   - Monte Carlo: `#include "barto_sutton_exercises/5_1/mc_fv_solution.h"` → `blackjack_main()`
   - TD: `#include "barto_sutton_exercises/5_1/td_solution.h"` → `blackjack_main()`

### Example

To switch to the Windy Gridworld environment with function approximation, edit `main.cpp`:

```cpp
#include "barto_sutton_exercises/6_9/fa_td_solution.h"

int main() {
    windygridworld_main();
    return 0;
}
```

Then rebuild: `cmake --build build` (or use `./run_rl.sh` which rebuilds automatically).
