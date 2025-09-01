# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -g -Wall -O2 -mavx2 -Iinclude -MMD

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin

# Source files
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
C_SOURCES   = $(wildcard $(SRC_DIR)/*.c)
SOURCES     = $(CPP_SOURCES) $(C_SOURCES)

# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o)
OBJECTS := $(OBJECTS:$(SRC_DIR)/%.c=$(SRC_DIR)/%.o)

# Dependency files (.d)
DEPS = $(OBJECTS:.o=.d)

# Output executable
TARGET = $(BIN_DIR)/main

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

# Compile .cpp and .c to .o, and generate .d files
$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create bin directory if it doesn't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Clean
clean:
	rm -f $(SRC_DIR)/*.o $(SRC_DIR)/*.d $(TARGET)

# Include dependency files, ignore errors if not found
-include $(DEPS)

.PHONY: all clean



