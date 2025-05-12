#!/usr/bin/env bash
set -e

# Minos-AI Project Test Script
# This script runs all tests for the Minos-AI project

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
SKIP_INTEGRATION=false
SKIP_E2E=false
VERBOSE=false
COVERAGE=false
UPDATE_SNAPSHOTS=false
BAIL=false
CI_MODE=false

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
    echo -e "${CYAN}${BOLD}Test All Script${NC}"
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
    echo "  -c, --coverage         Generate test coverage reports"
    echo "  -u, --update-snapshots Update test snapshots"
    echo "  -b, --bail             Stop running tests after the first failure"
    echo "  --ci                   Run in CI mode (skips user interaction)"
    echo "  --skip-contracts       Skip smart contract tests"
    echo "  --skip-backend         Skip backend tests"
    echo "  --skip-indexer         Skip indexer tests"
    echo "  --skip-sdk             Skip SDK tests"
    echo "  --skip-integration     Skip integration tests"
    echo "  --skip-e2e             Skip end-to-end tests"
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
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -u|--update-snapshots)
                UPDATE_SNAPSHOTS=true
                shift
                ;;
            -b|--bail)
                BAIL=true
                shift
                ;;
            --ci)
                CI_MODE=true
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
            --skip-integration)
                SKIP_INTEGRATION=true
                shift
                ;;
            --skip-e2e)
                SKIP_E2E=true
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
        echo -e "${GREEN}✓ Passed${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        echo -e "${RED}Error details:${NC}"
        cat "$output_file"
        rm "$output_file"
        
        if [ "$BAIL" = true ]; then
            echo -e "${RED}Stopping tests due to failure (--bail option).${NC}"
            exit 1
        fi
        
        return 1
    fi
    
    rm "$output_file"
    return 0
}

# Test smart contracts
test_contracts() {
    if [ "$SKIP_CONTRACTS" = true ]; then
        echo -e "${YELLOW}Skipping contract tests...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Testing Smart Contracts...${NC}"
    
    # Check if Rust and Solana are installed
    if ! command -v rustc &> /dev/null || ! command -v solana &> /dev/null || ! command -v anchor &> /dev/null; then
        echo -e "${YELLOW}Warning: Rust, Solana, or Anchor not installed. Skipping contract tests.${NC}"
        return 0
    fi
    
    # Check if contracts directory exists
    if [ ! -d "packages/contracts" ]; then
        echo -e "${YELLOW}Warning: Contracts directory not found. Skipping.${NC}"
        return 0
    fi
    
    # Change to contracts directory
    cd packages/contracts
    
    # Start local validator for testing if not in CI mode
    if [ "$CI_MODE" != true ]; then
        echo -e "${BLUE}Starting local Solana validator...${NC}"
        
        # Check if validator is already running
        if ! solana config get | grep -q "http://localhost:8899"; then
            solana config set --url localhost
        fi
        
        # Kill any existing validators
        pkill -f solana-test-validator || true
        
        # Start validator in background
        solana-test-validator --quiet &
        VALIDATOR_PID=$!
        
        # Wait for validator to start
        echo -e "${BLUE}Waiting for validator to start...${NC}"
        sleep 5
        
        # Verify validator is running
        if ! solana validators | grep -q "localhost"; then
            echo -e "${RED}Failed to start local validator${NC}"
            kill $VALIDATOR_PID 2>/dev/null || true
            cd ../..
            return 1
        fi
        
        echo -e "${GREEN}✓ Local validator started${NC}"
    fi
    
    # Run the contract tests
    echo -e "${BLUE}Running contract tests...${NC}"
    
    TEST_CMD="anchor test"
    
    if [ "$VERBOSE" = true ]; then
        TEST_CMD="$TEST_CMD --verbose"
    fi
    
    run_cmd "$TEST_CMD" "Running Anchor tests"
    TEST_RESULT=$?
    
    # Stop local validator if it was started
    if [ "$CI_MODE" != true ] && [ -n "$VALIDATOR_PID" ]; then
        echo -e "${BLUE}Stopping local validator...${NC}"
        kill $VALIDATOR_PID 2>/dev/null || true
    fi
    
    cd ../..
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Smart contract tests passed${NC}\n"
    else
        echo -e "${RED}✗ Smart contract tests failed${NC}\n"
        return 1
    fi
    
    return 0
}

# Test backend
test_backend() {
    if [ "$SKIP_BACKEND" = true ]; then
        echo -e "${YELLOW}Skipping backend tests...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Testing Backend...${NC}"
    
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
    
    # Build the backend if needed
    if [ ! -d "dist" ]; then
        run_cmd "npm run build" "Building backend"
    fi
    
    # Prepare test command
    TEST_CMD="npm test"
    
    if [ "$UPDATE_SNAPSHOTS" = true ]; then
        TEST_CMD="$TEST_CMD -- -u"
    fi
    
    if [ "$COVERAGE" = true ]; then
        TEST_CMD="npm run test:coverage"
    fi
    
    # Run the tests
    run_cmd "$TEST_CMD" "Running backend tests"
    TEST_RESULT=$?
    
    cd ../..
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Backend tests passed${NC}\n"
    else
        echo -e "${RED}✗ Backend tests failed${NC}\n"
        return 1
    fi
    
    return 0
}

# Test indexer
test_indexer() {
    if [ "$SKIP_INDEXER" = true ]; then
        echo -e "${YELLOW}Skipping indexer tests...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Testing Indexer...${NC}"
    
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
    
    # Build the indexer if needed
    if [ ! -d "dist" ]; then
        run_cmd "npm run build" "Building indexer"
    fi
    
    # Prepare test command
    TEST_CMD="npm test"
    
    if [ "$UPDATE_SNAPSHOTS" = true ]; then
        TEST_CMD="$TEST_CMD -- -u"
    fi
    
    if [ "$COVERAGE" = true ]; then
        TEST_CMD="npm run test:coverage"
    fi
    
    # Run the tests
    run_cmd "$TEST_CMD" "Running indexer tests"
    TEST_RESULT=$?
    
    cd ../..
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Indexer tests passed${NC}\n"
    else
        echo -e "${RED}✗ Indexer tests failed${NC}\n"
        return 1
    fi
    
    return 0
}

# Test SDK
test_sdk() {
    if [ "$SKIP_SDK" = true ]; then
        echo -e "${YELLOW}Skipping SDK tests...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Testing SDK...${NC}"
    
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
    
    # Build the SDK if needed
    if [ ! -d "dist" ]; then
        run_cmd "npm run build" "Building SDK"
    fi
    
    # Prepare test command
    TEST_CMD="npm test"
    
    if [ "$UPDATE_SNAPSHOTS" = true ]; then
        TEST_CMD="$TEST_CMD -- -u"
    fi
    
    if [ "$COVERAGE" = true ]; then
        TEST_CMD="npm run test:coverage"
    fi
    
    # Run the tests
    run_cmd "$TEST_CMD" "Running SDK tests"
    TEST_RESULT=$?
    
    cd ..
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ SDK tests passed${NC}\n"
    else
        echo -e "${RED}✗ SDK tests failed${NC}\n"
        return 1
    fi
    
    return 0
}

# Run integration tests
test_integration() {
    if [ "$SKIP_INTEGRATION" = true ]; then
        echo -e "${YELLOW}Skipping integration tests...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Running Integration Tests...${NC}"
    
    # Check for integration tests directory
    if [ ! -d "packages/contracts/tests" ]; then
        echo -e "${YELLOW}Warning: Integration tests directory not found. Skipping.${NC}"
        return 0
    fi
    
    # Prepare for testing
    echo -e "${BLUE}Preparing integration test environment...${NC}"
    
    # Start local validator for testing if not in CI mode
    if [ "$CI_MODE" != true ]; then
        echo -e "${BLUE}Starting local Solana validator...${NC}"
        
        # Check if validator is already running
        if ! solana config get | grep -q "http://localhost:8899"; then
            solana config set --url localhost
        fi
        
        # Kill any existing validators
        pkill -f solana-test-validator || true
        
        # Start validator in background
        solana-test-validator --quiet &
        VALIDATOR_PID=$!
        
        # Wait for validator to start
        echo -e "${BLUE}Waiting for validator to start...${NC}"
        sleep 5
        
        # Verify validator is running
        if ! solana validators | grep -q "localhost"; then
            echo -e "${RED}Failed to start local validator${NC}"
            kill $VALIDATOR_PID 2>/dev/null || true
            return 1
        fi
        
        echo -e "${GREEN}✓ Local validator started${NC}"
    fi
    
    # Run integration tests
    TEST_CMD="npm run test:integration"
    
    if [ "$VERBOSE" = true ]; then
        TEST_CMD="$TEST_CMD -- --verbose"
    fi
    
    run_cmd "$TEST_CMD" "Running integration tests"
    TEST_RESULT=$?
    
    # Stop local validator if it was started
    if [ "$CI_MODE" != true ] && [ -n "$VALIDATOR_PID" ]; then
        echo -e "${BLUE}Stopping local validator...${NC}"
        kill $VALIDATOR_PID 2>/dev/null || true
    fi
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Integration tests passed${NC}\n"
    else
        echo -e "${RED}✗ Integration tests failed${NC}\n"
        return 1
    fi
    
    return 0
}

# Run end-to-end tests
test_e2e() {
    if [ "$SKIP_E2E" = true ]; then
        echo -e "${YELLOW}Skipping end-to-end tests...${NC}"
        return 0
    fi
    
    echo -e "${CYAN}${BOLD}Running End-to-End Tests...${NC}"
    
    # Check for e2e tests directory
    if [ ! -d "tests/e2e" ]; then
        echo -e "${YELLOW}Warning: End-to-end tests directory not found. Skipping.${NC}"
        return 0
    fi
    
    # Run e2e tests
    TEST_CMD="npm run test:e2e"
    
    if [ "$VERBOSE" = true ]; then
        TEST_CMD="$TEST_CMD -- --verbose"
    fi
    
    run_cmd "$TEST_CMD" "Running end-to-end tests"
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ End-to-end tests passed${NC}\n"
    else
        echo -e "${RED}✗ End-to-end tests failed${NC}\n"
        return 1
    fi
    
    return 0
}

# Combine coverage reports
combine_coverage() {
    if [ "$COVERAGE" = true ]; then
        echo -e "${CYAN}${BOLD}Combining Coverage Reports...${NC}"
        
        # Create coverage directory if it doesn't exist
        mkdir -p coverage
        
        # Check for nyc (istanbul) coverage tool
        if ! command -v nyc &> /dev/null; then
            echo -e "${YELLOW}Warning: nyc not installed. Installing globally...${NC}"
            npm install -g nyc
        fi
        
        # Collect coverage reports
        cp -r packages/backend/coverage/. coverage/ 2>/dev/null || true
        cp -r packages/indexer/coverage/. coverage/ 2>/dev/null || true
        cp -r sdk/coverage/. coverage/ 2>/dev/null || true
        
        # Generate combined report
        if [ -d "coverage" ]; then
            run_cmd "nyc report --reporter=text --reporter=html --report-dir=coverage/combined" "Generating combined coverage report"
            echo -e "${GREEN}✓ Combined coverage report generated in coverage/combined${NC}\n"
        else
            echo -e "${YELLOW}Warning: No coverage reports found to combine.${NC}\n"
        fi
    fi
}

# Main function
main() {
    # Print the banner
    print_banner
    
    # Parse command-line arguments
    parse_args "$@"
    
    # Start time measurement
    START_TIME=$(date +%s)
    
    echo -e "${BLUE}${BOLD}Starting test process...${NC}\n"
    
    # Track test failures
    FAILURES=0
    
    # Test each component
    
    # Test smart contracts
    test_contracts
    if [ $? -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
    
    # Test SDK
    test_sdk
    if [ $? -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
    
    # Test backend
    test_backend
    if [ $? -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
    
    # Test indexer
    test_indexer
    if [ $? -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
    
    # Run integration tests
    test_integration
    if [ $? -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
    
    # Run end-to-end tests
    test_e2e
    if [ $? -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
    
    # Combine coverage reports if requested
    combine_coverage
    
    # End time measurement
    END_TIME=$(date +%s)
    TEST_DURATION=$((END_TIME - START_TIME))
    
    # Print summary
    echo -e "${BLUE}${BOLD}Test Summary${NC}"
    echo "=================================="
    echo -e "${CYAN}Test duration:${NC} ${TEST_DURATION} seconds"
    
    if [ $FAILURES -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ All tests passed!${NC}"
    else
        echo -e "${RED}${BOLD}✗ ${FAILURES} test suites failed!${NC}"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"