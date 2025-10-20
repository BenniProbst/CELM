# Makefile for CELM
# Wrapper around CMake build system

BUILD_DIR := build
INSTALL_PREFIX := /usr/local

# Default target
.PHONY: all
all: build

# Build the project
.PHONY: build
build:
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "Error: Build directory not found. Run ./configure first."; \
		exit 1; \
	fi
	@echo "Building CELM..."
	cmake --build $(BUILD_DIR) -j

# Run tests
.PHONY: test
test:
	@if [ ! -d "$(BUILD_DIR)" ]; then \
		echo "Error: Build directory not found. Run ./configure first."; \
		exit 1; \
	fi
	@echo "Running tests..."
	ctest --test-dir $(BUILD_DIR) --output-on-failure

# Install the binary
.PHONY: install
install: build
	@echo "Installing CELM..."
	cmake --install $(BUILD_DIR)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@if [ -d "$(BUILD_DIR)" ]; then \
		cmake --build $(BUILD_DIR) --target clean; \
	else \
		echo "Build directory not found, nothing to clean."; \
	fi

# Remove build directory completely
.PHONY: distclean
distclean:
	@echo "Removing build directory..."
	rm -rf $(BUILD_DIR)

# Uninstall (if install_manifest.txt exists)
.PHONY: uninstall
uninstall:
	@if [ -f "$(BUILD_DIR)/install_manifest.txt" ]; then \
		echo "Uninstalling CELM..."; \
		xargs rm -f < $(BUILD_DIR)/install_manifest.txt; \
		echo "Uninstall complete."; \
	else \
		echo "Error: install_manifest.txt not found. Cannot uninstall."; \
		exit 1; \
	fi

# Help target
.PHONY: help
help:
	@echo "CELM Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all         Build the project (default)"
	@echo "  build       Build the project"
	@echo "  test        Run tests"
	@echo "  install     Install the binary"
	@echo "  clean       Clean build artifacts"
	@echo "  distclean   Remove build directory completely"
	@echo "  uninstall   Uninstall the binary"
	@echo "  help        Display this help message"
	@echo ""
	@echo "Before running make, run ./configure to set up the build."
