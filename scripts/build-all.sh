#!/usr/bin/env bash
set -e

# Minos-AI Project Build Script
# This script builds all components of the Minos-AI project

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default settings
SKIP_CONTRACTS=false
SKIP_BACKEND=false
SKIP_INDEXER=false
SKIP_SDK=false
VERBOSE=false
CLEAN=false
PARALLEL=false

# Print banner
print_banner() {
    echo -e "${BLUE}${BOLD}"
    echo "███╗   ███╗██╗███╗   ██╗ ██████╗ ███████╗      █████╗ ██╗"
    echo "████╗ ████║██║████╗  ██║██╔═══██╗██╔════╝     ██╔══██╗██║"
    echo "██╔████╔██║██║██╔██╗ ██║██║   ██║███████╗     ███████║██║"
    echo "██║╚██╔╝██║██║██║╚██╗██║██║   ██║╚════██║     ██╔══██║██║"
    echo "██║ ╚═╝ ██║██║██║ ╚████║╚██████╔╝███████║     ██║  ██║██║"
    echo "╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝     ╚═╝  ╚═╝╚═╝"
    echo -e "${NC}"
    echo -e "${CYAN}${BOLD}Build All Script${NC}"
    echo "=================================="
    echo ""
}

# Print usage information
print_usage() {
    echo -e "${BOLD}Usage:${NC} $0 [options]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo "  -h, --help             Show this help message"
    echo "  -v, --verbose          Enable verbose output"
    echo "  -c, --clean            Clean before building"
    echo "  -p, --parallel         Build components in parallel (faster)"
    echo "  --skip-contracts       Skip building smart contracts"
    echo "  --skip-backend         Skip building backend"
    echo "  --skip-indexer         Skip building indexer"
    echo "  --skip-sdk             Skip building SDK"
    echo ""
}

# Parse command-line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                print_usage
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -c|--clean)
                CLEAN=true
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            --skip-contracts)
                SKIP_CONTRACTS=true
                shift
                ;;
            --skip-backend)
                SKIP_BACKEND=true
                shift
                ;;
            --skip-indexer)
                SKIP_INDEXER=true
                shift
                ;;
            --skip-sdk)
                SKIP_SDK=true
                shift
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Run a command with proper output handling
run_cmd() {
    local cmd=$1
    local desc=$2
    local output_file=$(mktemp)
    
    echo -ne "${BLUE}${desc}...${NC} "
    
    if [ "$VERBOSE" = true ]; then
        echo ""
        eval "$cmd" | tee "$output_file"
        local result=$?
    else
        eval "$cmd" > "$output_file" 2>&1
        local result=$?
    fi
    
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✓ Done${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        echo -e "${RED}Error details:${NC}"
        cat "$output_file"
        rm "$output_file"
        return 1
    fi
    
    rm "$output_file"
    return 0
}

# Clean build artifacts
clean_artifacts() {
    if [ "$CLEAN" = true ]; then
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        
        # Clean SDK
        if [ -d "sdk" ] && [ -d "sdk/dist" ]; then
            run_cmd "rm -rf sdk/dist" "Cleaning SDK"
        fi
        
        # Clean backend
        if [ -d "packages/backend" ] && [ -d "packages/backend/dist" ]; then
            run_cmd "rm -rf packages/backend/dist" "Cleaning backend"
        fi
        
        # Clean indexer
        if [ -d "packages/indexer" ] && [ -d "packages/indexer/dist" ]; then
            run_cmd "rm -rf packages/indexer/dist" "Cleaning indexer"
        fi
        
        # Clean contracts
        if [ -d "packages/contracts" ] && [ -d "packages/contracts/target" ]; then
            run_cmd "(cd packages/contracts && cargo clean)" "Cleaning contracts"
        fi
        
        echo -e "${GREEN}✓ All artifacts cleaned${NC}\n"
    fi
}

# Build smart contracts
build_contracts() {
    if [ "$SKIP_CONTRACTS" = true ]; then
        echo -e "${YELLOW}Skipping contracts build...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Building Smart Contracts...${NC}"
    
    # Check if Rust and Solana are installed
    if ! command -v rustc &> /dev/null || ! command -v solana &> /dev/null || ! command -v anchor &> /dev/null; then
        echo -e "${YELLOW}Warning: Rust, Solana, or Anchor not installed. Skipping contracts build.${NC}"
        return 0
    fi
    
    # Check if contracts directory exists
    if [ ! -d "packages/contracts" ]; then
        echo -e "${YELLOW}Warning: Contracts directory not found. Skipping.${NC}"
        return 0
    fi
    
    # Build the smart contracts
    cd packages/contracts
    
    # Build with Anchor
    run_cmd "anchor build" "Building Anchor programs"
    
    # Verify build output
    if [ -d "target/deploy" ]; then
        echo -e "${GREEN}✓ Contracts built successfully${NC}"
        
        # Show program IDs
        echo -e "${CYAN}Program IDs:${NC}"
        for program in target/deploy/*.so; do
            if [ -f "$program" ]; then
                program_name=$(basename "$program" .so)
                echo -e "${CYAN}  - ${program_name}:${NC} $(solana-keygen pubkey "target/deploy/${program_name}-keypair.json")"
            fi
        done
    else
        echo -e "${RED}✗ No build artifacts found${NC}"
        cd ../..
        return 1
    fi
    
    cd ../..
    echo -e "${GREEN}✓ Smart contracts build complete${NC}\n"
}

# Build backend
build_backend() {
    if [ "$SKIP_BACKEND" = true ]; then
        echo -e "${YELLOW}Skipping backend build...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Building Backend...${NC}"
    
    # Check if backend directory exists
    if [ ! -d "packages/backend" ]; then
        echo -e "${YELLOW}Warning: Backend directory not found. Skipping.${NC}"
        return 0
    fi
    
    # Change to backend directory
    cd packages/backend
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ] || [ ! -f "node_modules/.package-lock.json" ]; then
        run_cmd "npm install" "Installing backend dependencies"
    fi
    
    # Build the backend
    run_cmd "npm run build" "Building backend"
    
    # Verify build output
    if [ -d "dist" ]; then
        echo -e "${GREEN}✓ Backend built successfully${NC}"
    else
        echo -e "${RED}✗ No build artifacts found${NC}"
        cd ../..
        return 1
    fi
    
    cd ../..
    echo -e "${GREEN}✓ Backend build complete${NC}\n"
}

# Build indexer
build_indexer() {
    if [ "$SKIP_INDEXER" = true ]; then
        echo -e "${YELLOW}Skipping indexer build...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Building Indexer...${NC}"
    
    # Check if indexer directory exists
    if [ ! -d "packages/indexer" ]; then
        echo -e "${YELLOW}Warning: Indexer directory not found. Skipping.${NC}"
        return 0
    fi
    
    # Change to indexer directory
    cd packages/indexer
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ] || [ ! -f "node_modules/.package-lock.json" ]; then
        run_cmd "npm install" "Installing indexer dependencies"
    fi
    
    # Build the indexer
    run_cmd "npm run build" "Building indexer"
    
    # Verify build output
    if [ -d "dist" ]; then
        echo -e "${GREEN}✓ Indexer built successfully${NC}"
    else
        echo -e "${RED}✗ No build artifacts found${NC}"
        cd ../..
        return 1
    fi
    
    cd ../..
    echo -e "${GREEN}✓ Indexer build complete${NC}\n"
}

# Build SDK
build_sdk() {
    if [ "$SKIP_SDK" = true ]; then
        echo -e "${YELLOW}Skipping SDK build...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Building SDK...${NC}"
    
    # Check if SDK directory exists
    if [ ! -d "sdk" ]; then
        echo -e "${YELLOW}Warning: SDK directory not found. Skipping.${NC}"
        return 0
    fi
    
    # Change to SDK directory
    cd sdk
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ] || [ ! -f "node_modules/.package-lock.json" ]; then
        run_cmd "npm install" "Installing SDK dependencies"
    fi
    
    # Build the SDK
    run_cmd "npm run build" "Building SDK"
    
    # Verify build output
    if [ -d "dist" ]; then
        echo -e "${GREEN}✓ SDK built successfully${NC}"
    else
        echo -e "${RED}✗ No build artifacts found${NC}"
        cd ..
        return 1
    fi
    
    cd ..
    echo -e "${GREEN}✓ SDK build complete${NC}\n"
}

# Build all components in parallel
build_parallel() {
    echo -e "${BLUE}${BOLD}Building all components in parallel...${NC}\n"
    
    # Create temporary directory for logs
    mkdir -p temp_logs
    
    # Start builds in parallel
    if [ "$SKIP_CONTRACTS" != true ]; then
        (build_contracts > temp_logs/contracts.log 2>&1) &
        CONTRACT_PID=$!
        echo -e "${CYAN}Started contracts build (PID: $CONTRACT_PID)${NC}"
    fi
    
    if [ "$SKIP_BACKEND" != true ]; then
        (build_backend > temp_logs/backend.log 2>&1) &
        BACKEND_PID=$!
        echo -e "${CYAN}Started backend build (PID: $BACKEND_PID)${NC}"
    fi
    
    if [ "$SKIP_INDEXER" != true ]; then
        (build_indexer > temp_logs/indexer.log 2>&1) &
        INDEXER_PID=$!
        echo -e "${CYAN}Started indexer build (PID: $INDEXER_PID)${NC}"
    fi
    
    if [ "$SKIP_SDK" != true ]; then
        (build_sdk > temp_logs/sdk.log 2>&1) &
        SDK_PID=$!
        echo -e "${CYAN}Started SDK build (PID: $SDK_PID)${NC}"
    fi
    
    # Wait for all builds to complete
    echo -e "${BLUE}Waiting for all builds to complete...${NC}"
    
    FAILED=0
    
    if [ "$SKIP_CONTRACTS" != true ]; then
        wait $CONTRACT_PID
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Contracts build failed${NC}"
            FAILED=1
        else
            echo -e "${GREEN}✓ Contracts build completed successfully${NC}"
        fi
    fi
    
    if [ "$SKIP_BACKEND" != true ]; then
        wait $BACKEND_PID
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Backend build failed${NC}"
            FAILED=1
        else
            echo -e "${GREEN}✓ Backend build completed successfully${NC}"
        fi
    fi
    
    if [ "$SKIP_INDEXER" != true ]; then
        wait $INDEXER_PID
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ Indexer build failed${NC}"
            FAILED=1
        else
            echo -e "${GREEN}✓ Indexer build completed successfully${NC}"
        fi
    fi
    
    if [ "$SKIP_SDK" != true ]; then
        wait $SDK_PID
        if [ $? -ne 0 ]; then
            echo -e "${RED}✗ SDK build failed${NC}"
            FAILED=1
        else
            echo -e "${GREEN}✓ SDK build completed successfully${NC}"
        fi
    fi
    
    # If verbose, show logs from failed builds
    if [ "$VERBOSE" = true ] && [ $FAILED -ne 0 ]; then
        echo -e "${RED}Build logs for failed components:${NC}"
        
        if [ "$SKIP_CONTRACTS" != true ] && [ ! -f temp_logs/contracts.log ]; then
            echo -e "${RED}Contracts build log:${NC}"
            cat temp_logs/contracts.log
            echo ""
        fi
        
        if [ "$SKIP_BACKEND" != true ] && [ ! -f temp_logs/backend.log ]; then
            echo -e "${RED}Backend build log:${NC}"
            cat temp_logs/backend.log
            echo ""
        fi
        
        if [ "$SKIP_INDEXER" != true ] && [ ! -f temp_logs/indexer.log ]; then
            echo -e "${RED}Indexer build log:${NC}"
            cat temp_logs/indexer.log
            echo ""
        fi
        
        if [ "$SKIP_SDK" != true ] && [ ! -f temp_logs/sdk.log ]; then
            echo -e "${RED}SDK build log:${NC}"
            cat temp_logs/sdk.log
            echo ""
        fi
    fi
    
    # Remove temporary logs directory
    rm -rf temp_logs
    
    if [ $FAILED -ne 0 ]; then
        return 1
    fi
    
    return 0
}

# Build all components sequentially
build_sequential() {
    # Build smart contracts
    build_contracts
    
    # Build SDK
    build_sdk
    
    # Build backend
    build_backend
    
    # Build indexer
    build_indexer
}

# Main function
main() {
    # Print the banner
    print_banner
    
    # Parse command-line arguments
    parse_args "$@"
    
    # Start time measurement
    START_TIME=$(date +%s)
    
    echo -e "${BLUE}${BOLD}Starting build process...${NC}\n"
    
    # Clean artifacts if requested
    clean_artifacts
    
    # Build all components
    if [ "$PARALLEL" = true ]; then
        build_parallel
        BUILD_RESULT=$?
    else
        build_sequential
        BUILD_RESULT=$?
    fi
    
    # End time measurement
    END_TIME=$(date +%s)
    BUILD_DURATION=$((END_TIME - START_TIME))
    
    # Print summary
    echo -e "${BLUE}${BOLD}Build Summary${NC}"
    echo "=================================="
    echo -e "${CYAN}Build duration:${NC} ${BUILD_DURATION} seconds"
    
    if [ $BUILD_RESULT -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ All components built successfully!${NC}"
    else
        echo -e "${RED}${BOLD}✗ Build failed!${NC}"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"