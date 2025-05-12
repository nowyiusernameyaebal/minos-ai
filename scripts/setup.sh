#!/usr/bin/env bash
set -e

# Minos-AI Project Setup Script
# This script sets up the development environment for the Minos-AI project.
# It checks for required dependencies, installs missing tools, configures the environment,
# and builds the project components.

# Version
VERSION="1.0.0"

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
DEFAULT_NODE_VERSION="18.0.0"
DEFAULT_SOLANA_VERSION="1.16.13"
DEFAULT_ANCHOR_VERSION="0.28.0"
DEFAULT_ENVIRONMENT="devnet" # Can be: localnet, devnet, testnet, mainnet
INSTALL_DOCKER=false
INSTALL_DATABASE=false
SKIP_RUST=false
SKIP_SOLANA=false
SKIP_ANCHOR=false
SKIP_BUILD=false
VERBOSE=false
SKIP_CONFIRM=false
LOG_FILE="setup_log.txt"

# Print banner
print_banner() {
    clear
    echo -e "${BLUE}${BOLD}"
    echo " ███╗   ███╗██╗███╗   ██╗ ██████╗ ███████╗      █████╗ ██╗"
    echo " ████╗ ████║██║████╗  ██║██╔═══██╗██╔════╝     ██╔══██╗██║"
    echo " ██╔████╔██║██║██╔██╗ ██║██║   ██║███████╗     ███████║██║"
    echo " ██║╚██╔╝██║██║██║╚██╗██║██║   ██║╚════██║     ██╔══██║██║"
    echo " ██║ ╚═╝ ██║██║██║ ╚████║╚██████╔╝███████║     ██║  ██║██║"
    echo " ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝     ╚═╝  ╚═╝╚═╝"
    echo -e "${NC}"
    echo -e " ${CYAN}${BOLD}Project Setup Script v${VERSION}${NC}"
    echo -e " ${CYAN}${BOLD}==========================================${NC}"
    echo ""
}

# Print usage information
print_usage() {
    echo -e "${BOLD}Usage:${NC} $0 [options]"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo "  -h, --help                Show this help message"
    echo "  -v, --verbose             Enable verbose output"
    echo "  -y, --yes                 Skip all confirmation prompts (assume yes)"
    echo "  -e, --environment ENV     Set Solana environment (localnet, devnet, testnet, mainnet)"
    echo "  --node-version VERSION    Specify required Node.js version (default: ${DEFAULT_NODE_VERSION})"
    echo "  --solana-version VERSION  Specify Solana version to install (default: ${DEFAULT_SOLANA_VERSION})"
    echo "  --anchor-version VERSION  Specify Anchor version to install (default: ${DEFAULT_ANCHOR_VERSION})"
    echo "  --with-docker             Install and configure Docker"
    echo "  --with-database           Install and configure local database"
    echo "  --skip-rust               Skip Rust installation and checks"
    echo "  --skip-solana             Skip Solana installation and configuration"
    echo "  --skip-anchor             Skip Anchor installation"
    echo "  --skip-build              Skip project build step"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0 --environment devnet --with-docker"
    echo "  $0 -y --skip-rust --skip-solana"
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
            -y|--yes)
                SKIP_CONFIRM=true
                shift
                ;;
            -e|--environment)
                DEFAULT_ENVIRONMENT="$2"
                shift 2
                ;;
            --node-version)
                DEFAULT_NODE_VERSION="$2"
                shift 2
                ;;
            --solana-version)
                DEFAULT_SOLANA_VERSION="$2"
                shift 2
                ;;
            --anchor-version)
                DEFAULT_ANCHOR_VERSION="$2"
                shift 2
                ;;
            --with-docker)
                INSTALL_DOCKER=true
                shift
                ;;
            --with-database)
                INSTALL_DATABASE=true
                shift
                ;;
            --skip-rust)
                SKIP_RUST=true
                shift
                ;;
            --skip-solana)
                SKIP_SOLANA=true
                shift
                ;;
            --skip-anchor)
                SKIP_ANCHOR=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
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

# Initialize logging
init_logging() {
    # Clear previous log file
    echo "" > "$LOG_FILE"
    echo "Minos-AI Setup Log - $(date)" >> "$LOG_FILE"
    echo "================================" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    log "Setup script started"
}

# Logging function
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" >> "$LOG_FILE"
    
    if [ "$VERBOSE" = true ]; then
        echo -e "$1"
    fi
}

# Confirm action
confirm() {
    if [ "$SKIP_CONFIRM" = true ]; then
        return 0
    fi
    
    read -r -p "$1 [Y/n] " response
    response=${response,,} # Convert to lowercase
    if [[ $response =~ ^(no|n)$ ]]; then
        return 1
    else
        return 0
    fi
}

# Check for required tools
check_requirements() {
    echo -e "${BLUE}${BOLD}Checking required tools...${NC}"
    log "Checking required tools"
    
    # Check for Git
    if ! command -v git &> /dev/null; then
        echo -e "${RED}Error: Git is not installed.${NC}"
        echo "Please install Git from https://git-scm.com/"
        log "ERROR: Git not installed"
        exit 1
    else
        GIT_VERSION=$(git --version)
        echo -e "${GREEN}✓ ${GIT_VERSION} is installed${NC}"
        log "Git version: $GIT_VERSION"
    fi
    
    # Check for Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${RED}Error: Node.js is not installed.${NC}"
        echo "Please install Node.js v${DEFAULT_NODE_VERSION} or higher from https://nodejs.org"
        log "ERROR: Node.js not installed"
        exit 1
    else
        NODE_VERSION=$(node -v)
        echo -e "${GREEN}✓ Node.js ${NODE_VERSION} is installed${NC}"
        log "Node.js version: $NODE_VERSION"
        
        # Check Node.js version
        NODE_VERSION_CLEAN=$(echo "$NODE_VERSION" | sed 's/v//')
        if ! check_version "$NODE_VERSION_CLEAN" "$DEFAULT_NODE_VERSION"; then
            echo -e "${YELLOW}Warning: Node.js version ${NODE_VERSION} is lower than recommended (${DEFAULT_NODE_VERSION}).${NC}"
            echo "Some features may not work correctly."
            log "WARNING: Node.js version lower than recommended"
            
            if confirm "Would you like to upgrade Node.js?"; then
                echo "Please upgrade Node.js manually to v${DEFAULT_NODE_VERSION} or higher."
                echo "Visit https://nodejs.org for installation instructions."
            fi
        fi
    fi
    
    # Check for npm
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}Error: npm is not installed.${NC}"
        echo "Please install npm (usually comes with Node.js)"
        log "ERROR: npm not installed"
        exit 1
    else
        NPM_VERSION=$(npm -v)
        echo -e "${GREEN}✓ npm ${NPM_VERSION} is installed${NC}"
        log "npm version: $NPM_VERSION"
    fi
    
    # Check for Yarn (optional)
    if command -v yarn &> /dev/null; then
        YARN_VERSION=$(yarn -v)
        echo -e "${GREEN}✓ Yarn ${YARN_VERSION} is installed${NC}"
        log "Yarn version: $YARN_VERSION"
    else
        echo -e "${YELLOW}Note: Yarn is not installed.${NC}"
        echo "npm will be used for package management."
        log "Yarn not installed, using npm"
    fi
    
    # Check for Rust
    if [ "$SKIP_RUST" = false ]; then
        if ! command -v rustc &> /dev/null; then
            echo -e "${YELLOW}Warning: Rust is not installed.${NC}"
            log "WARNING: Rust not installed"
            
            if confirm "Would you like to install Rust? (recommended for smart contract development)"; then
                echo "Installing Rust..."
                log "Installing Rust"
                curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                source "$HOME/.cargo/env"
                echo -e "${GREEN}✓ Rust installed successfully${NC}"
                log "Rust installed successfully"
                
                # Install additional Rust components
                echo "Installing additional Rust components..."
                rustup component add rustfmt
                rustup component add clippy
                rustup target add wasm32-unknown-unknown
                log "Installed additional Rust components"
            else
                echo -e "${YELLOW}Skipping Rust installation.${NC}"
                echo "Note: You won't be able to build the smart contracts without Rust."
                log "Skipped Rust installation"
            fi
        else
            RUST_VERSION=$(rustc --version)
            echo -e "${GREEN}✓ ${RUST_VERSION} is installed${NC}"
            log "Rust version: $RUST_VERSION"
        fi
    else
        echo -e "${YELLOW}Skipping Rust check as requested.${NC}"
        log "Skipped Rust check (--skip-rust)"
    fi
    
    # Check for Solana
    if [ "$SKIP_SOLANA" = false ]; then
        if ! command -v solana &> /dev/null; then
            echo -e "${YELLOW}Warning: Solana CLI is not installed.${NC}"
            log "WARNING: Solana CLI not installed"
            
            if confirm "Would you like to install Solana CLI? (recommended for smart contract deployment)"; then
                echo "Installing Solana CLI..."
                log "Installing Solana CLI v${DEFAULT_SOLANA_VERSION}"
                sh -c "$(curl -sSfL https://release.solana.com/v${DEFAULT_SOLANA_VERSION}/install)"
                export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
                echo 'export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"' >> ~/.profile
                echo 'export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"' >> ~/.bashrc
                echo -e "${GREEN}✓ Solana CLI installed successfully${NC}"
                log "Solana CLI installed successfully"
            else
                echo -e "${YELLOW}Skipping Solana CLI installation.${NC}"
                echo "Note: You won't be able to deploy smart contracts without Solana CLI."
                log "Skipped Solana CLI installation"
            fi
        else
            SOLANA_VERSION=$(solana --version | head -n 1)
            echo -e "${GREEN}✓ ${SOLANA_VERSION} is installed${NC}"
            log "Solana version: $SOLANA_VERSION"
        fi
    else
        echo -e "${YELLOW}Skipping Solana check as requested.${NC}"
        log "Skipped Solana check (--skip-solana)"
    fi
    
    # Check for Anchor
    if [ "$SKIP_ANCHOR" = false ] && [ "$SKIP_SOLANA" = false ]; then
        if ! command -v anchor &> /dev/null; then
            echo -e "${YELLOW}Warning: Anchor CLI is not installed.${NC}"
            log "WARNING: Anchor CLI not installed"
            
            if confirm "Would you like to install Anchor CLI? (required for smart contract development)"; then
                echo "Installing Anchor CLI..."
                log "Installing Anchor CLI v${DEFAULT_ANCHOR_VERSION}"
                # Install cargo-dependencies
                cargo install --git https://github.com/project-serum/anchor avm --locked --force
                avm install ${DEFAULT_ANCHOR_VERSION}
                avm use ${DEFAULT_ANCHOR_VERSION}
                echo -e "${GREEN}✓ Anchor CLI installed successfully${NC}"
                log "Anchor CLI installed successfully"
            else
                echo -e "${YELLOW}Skipping Anchor CLI installation.${NC}"
                echo "Note: You won't be able to build and test the smart contracts without Anchor CLI."
                log "Skipped Anchor CLI installation"
            fi
        else
            ANCHOR_VERSION=$(anchor --version)
            echo -e "${GREEN}✓ ${ANCHOR_VERSION} is installed${NC}"
            log "Anchor version: $ANCHOR_VERSION"
        fi
    else
        echo -e "${YELLOW}Skipping Anchor check as requested.${NC}"
        log "Skipped Anchor check (--skip-anchor or --skip-solana)"
    fi
    
    # Check for Docker (optional)
    if [ "$INSTALL_DOCKER" = true ]; then
        if ! command -v docker &> /dev/null; then
            echo -e "${YELLOW}Warning: Docker is not installed.${NC}"
            log "WARNING: Docker not installed"
            
            if confirm "Would you like to install Docker?"; then
                echo "Installing Docker..."
                log "Installing Docker"
                
                # Detect OS
                if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                    # Install Docker on Linux
                    curl -fsSL https://get.docker.com -o get-docker.sh
                    sudo sh get-docker.sh
                    sudo usermod -aG docker "$(whoami)"
                    echo -e "${GREEN}✓ Docker installed successfully${NC}"
                    echo -e "${YELLOW}Note: You may need to log out and back in for Docker permissions to take effect.${NC}"
                    log "Docker installed successfully on Linux"
                elif [[ "$OSTYPE" == "darwin"* ]]; then
                    # Install Docker on macOS
                    echo "Please install Docker Desktop for Mac manually from https://www.docker.com/products/docker-desktop"
                    log "Directed to install Docker Desktop for Mac manually"
                else
                    echo -e "${YELLOW}Automatic Docker installation not supported for this OS.${NC}"
                    echo "Please install Docker manually from https://www.docker.com/get-started"
                    log "Automatic Docker installation not supported for this OS"
                fi
            else
                echo -e "${YELLOW}Skipping Docker installation.${NC}"
                log "Skipped Docker installation"
            fi
        else
            DOCKER_VERSION=$(docker --version)
            echo -e "${GREEN}✓ ${DOCKER_VERSION} is installed${NC}"
            log "Docker version: $DOCKER_VERSION"
            
            # Check Docker Compose
            if command -v docker-compose &> /dev/null; then
                DOCKER_COMPOSE_VERSION=$(docker-compose --version)
                echo -e "${GREEN}✓ ${DOCKER_COMPOSE_VERSION} is installed${NC}"
                log "Docker Compose version: $DOCKER_COMPOSE_VERSION"
            else
                echo -e "${YELLOW}Warning: Docker Compose is not installed.${NC}"
                log "WARNING: Docker Compose not installed"
                
                if confirm "Would you like to install Docker Compose?"; then
                    echo "Installing Docker Compose..."
                    log "Installing Docker Compose"
                    
                    # Install Docker Compose
                    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
                    sudo chmod +x /usr/local/bin/docker-compose
                    echo -e "${GREEN}✓ Docker Compose installed successfully${NC}"
                    log "Docker Compose installed successfully"
                fi
            fi
        fi
    else
        if ! command -v docker &> /dev/null; then
            echo -e "${YELLOW}Note: Docker is not installed.${NC}"
            echo "Docker is optional but recommended for running local services."
            echo "You can install Docker from https://docs.docker.com/get-docker/"
            log "Docker not installed (optional)"
        else
            DOCKER_VERSION=$(docker --version)
            echo -e "${GREEN}✓ ${DOCKER_VERSION} is installed${NC}"
            log "Docker version: $DOCKER_VERSION"
        fi
    fi
    
    # Install Database if requested
    if [ "$INSTALL_DATABASE" = true ]; then
        install_database
    fi
    
    echo -e "${GREEN}All required tools checked!${NC}\n"
    log "All required tools checked"
}

# Install database (PostgreSQL)
install_database() {
    echo -e "${BLUE}Setting up database...${NC}"
    log "Setting up database"
    
    # Check if PostgreSQL is already installed
    if command -v psql &> /dev/null; then
        POSTGRES_VERSION=$(psql --version)
        echo -e "${GREEN}✓ ${POSTGRES_VERSION} is already installed${NC}"
        log "PostgreSQL already installed: $POSTGRES_VERSION"
    else
        echo -e "${YELLOW}PostgreSQL is not installed.${NC}"
        log "PostgreSQL not installed"
        
        if confirm "Would you like to install PostgreSQL?"; then
            echo "Installing PostgreSQL..."
            log "Installing PostgreSQL"
            
            # Detect OS
            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                # Install PostgreSQL on Linux
                sudo apt-get update
                sudo apt-get install -y postgresql postgresql-contrib
                echo -e "${GREEN}✓ PostgreSQL installed successfully${NC}"
                log "PostgreSQL installed successfully on Linux"
                
                # Start PostgreSQL service
                sudo systemctl enable postgresql
                sudo systemctl start postgresql
                echo -e "${GREEN}✓ PostgreSQL service started${NC}"
                log "PostgreSQL service started"
                
                # Create database user and database
                if confirm "Would you like to create a database and user for the project?"; then
                    echo "Creating database user and database..."
                    log "Creating database user and database"
                    
                    read -r -p "Enter database name [minos_ai]: " DB_NAME
                    DB_NAME=${DB_NAME:-minos_ai}
                    
                    read -r -p "Enter database user [minos_user]: " DB_USER
                    DB_USER=${DB_USER:-minos_user}
                    
                    read -r -s -p "Enter database password: " DB_PASSWORD
                    echo
                    
                    # Create user and database
                    sudo -u postgres psql -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
                    sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;"
                    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
                    
                    echo -e "${GREEN}✓ Database and user created successfully${NC}"
                    log "Database and user created successfully"
                    
                    # Update .env file with database credentials
                    if [ -f .env ]; then
                        sed -i "s/DB_NAME=.*/DB_NAME=$DB_NAME/" .env
                        sed -i "s/DB_USER=.*/DB_USER=$DB_USER/" .env
                        sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=$DB_PASSWORD/" .env
                        sed -i "s/DB_HOST=.*/DB_HOST=localhost/" .env
                        sed -i "s/DB_PORT=.*/DB_PORT=5432/" .env
                        echo -e "${GREEN}✓ Updated .env file with database credentials${NC}"
                        log "Updated .env file with database credentials"
                    fi
                fi
            elif [[ "$OSTYPE" == "darwin"* ]]; then
                # Install PostgreSQL on macOS using Homebrew
                if ! command -v brew &> /dev/null; then
                    echo -e "${YELLOW}Homebrew is not installed. Installing Homebrew...${NC}"
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                    log "Installed Homebrew"
                fi
                
                brew install postgresql
                brew services start postgresql
                echo -e "${GREEN}✓ PostgreSQL installed and started successfully${NC}"
                log "PostgreSQL installed successfully on macOS"
                
                # Create database and user
                if confirm "Would you like to create a database and user for the project?"; then
                    echo "Creating database user and database..."
                    log "Creating database user and database"
                    
                    read -r -p "Enter database name [minos_ai]: " DB_NAME
                    DB_NAME=${DB_NAME:-minos_ai}
                    
                    read -r -p "Enter database user [minos_user]: " DB_USER
                    DB_USER=${DB_USER:-minos_user}
                    
                    read -r -s -p "Enter database password: " DB_PASSWORD
                    echo
                    
                    # Create user and database
                    psql postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
                    psql postgres -c "CREATE DATABASE $DB_NAME;"
                    psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
                    
                    echo -e "${GREEN}✓ Database and user created successfully${NC}"
                    log "Database and user created successfully"
                    
                    # Update .env file with database credentials
                    if [ -f .env ]; then
                        sed -i '' "s/DB_NAME=.*/DB_NAME=$DB_NAME/" .env
                        sed -i '' "s/DB_USER=.*/DB_USER=$DB_USER/" .env
                        sed -i '' "s/DB_PASSWORD=.*/DB_PASSWORD=$DB_PASSWORD/" .env
                        sed -i '' "s/DB_HOST=.*/DB_HOST=localhost/" .env
                        sed -i '' "s/DB_PORT=.*/DB_PORT=5432/" .env
                        echo -e "${GREEN}✓ Updated .env file with database credentials${NC}"
                        log "Updated .env file with database credentials"
                    fi
                fi
            else
                echo -e "${YELLOW}Automatic PostgreSQL installation not supported for this OS.${NC}"
                echo "Please install PostgreSQL manually from https://www.postgresql.org/download/"
                log "Automatic PostgreSQL installation not supported for this OS"
            fi
        else
            echo -e "${YELLOW}Skipping PostgreSQL installation.${NC}"
            log "Skipped PostgreSQL installation"
        fi
    fi
}

# Configure environment
configure_environment() {
    echo -e "${BLUE}${BOLD}Configuring environment...${NC}"
    log "Configuring environment"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "Creating .env file..."
        log "Creating .env file"
        
        if [ -f .env.example ]; then
            cp .env.example .env
            echo -e "${GREEN}✓ Created .env file from example${NC}"
            log "Created .env file from example"
        else
            echo "Creating default .env file..."
            cat > .env << EOL
# Minos-AI Environment Configuration

# General
NODE_ENV=development
LOG_LEVEL=info

# Solana
SOLANA_RPC_URL=$(get_solana_url "$DEFAULT_ENVIRONMENT")
SOLANA_NETWORK=$DEFAULT_ENVIRONMENT
SOLANA_COMMITMENT=confirmed

# Wallet (Generate a new Solana keypair or add your own)
# SOLANA_PRIVATE_KEY=

# Program IDs
VAULT_PROGRAM_ID=$(get_program_id "vault" "$DEFAULT_ENVIRONMENT")
AI_AGENT_PROGRAM_ID=$(get_program_id "ai-agent" "$DEFAULT_ENVIRONMENT")
GOVERNANCE_PROGRAM_ID=$(get_program_id "governance" "$DEFAULT_ENVIRONMENT")

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=minos_ai
DB_USER=minos_user
DB_PASSWORD=
DB_SSL=false

# API
PORT=3000
API_PREFIX=/api
RATE_LIMIT_WINDOW=15
RATE_LIMIT_MAX=100
API_DOCS_PATH=/docs

# JWT
JWT_SECRET=$(generate_secret)
JWT_EXPIRES_IN=7d

# CORS
CORS_ORIGIN=*

# Services
INDEXER_ENABLED=true
BACKEND_ENABLED=true
EOL
            echo -e "${GREEN}✓ Created default .env file${NC}"
            log "Created default .env file"
        fi
        
        echo -e "${YELLOW}Note: Please edit the .env file with your specific configuration.${NC}"
    else
        echo -e "${GREEN}✓ .env file already exists${NC}"
        log ".env file already exists"
    fi
    
    # Configure Solana if installed and not skipped
    if command -v solana &> /dev/null && [ "$SKIP_SOLANA" = false ]; then
        echo "Configuring Solana..."
        log "Configuring Solana for $DEFAULT_ENVIRONMENT"
        
        # Check if a Solana keypair exists
        if [ ! -f ~/.config/solana/id.json ]; then
            echo "Generating new Solana keypair..."
            log "Generating new Solana keypair"
            solana-keygen new --no-bip39-passphrase -o ~/.config/solana/id.json --silent
            echo -e "${GREEN}✓ Generated new Solana keypair${NC}"
            log "Generated new Solana keypair"
        else
            echo -e "${GREEN}✓ Solana keypair already exists${NC}"
            log "Solana keypair already exists"
        fi
        
        # Set Solana configuration
        echo "Setting Solana config to $DEFAULT_ENVIRONMENT..."
        log "Setting Solana config to $DEFAULT_ENVIRONMENT"
        
        # Set correct RPC URL based on environment
        SOLANA_URL=$(get_solana_url "$DEFAULT_ENVIRONMENT")
        solana config set --url "$SOLANA_URL"
        echo -e "${GREEN}✓ Solana configured to use $DEFAULT_ENVIRONMENT${NC}"
        log "Solana configured to use $DEFAULT_ENVIRONMENT"
        
        # Add keypair to .env file
        if confirm "Would you like to add your Solana keypair to .env?"; then
            echo "Adding Solana keypair to .env file..."
            log "Adding Solana keypair to .env file"
            
            KEYPAIR_PATH=~/.config/solana/id.json
            PRIVATE_KEY=$(cat "$KEYPAIR_PATH" | jq -r 'tostring')
            
            # Update .env file
            sed -i.bak "s/# SOLANA_PRIVATE_KEY=/SOLANA_PRIVATE_KEY=$PRIVATE_KEY/" .env
            rm .env.bak
            
            echo -e "${GREEN}✓ Added Solana keypair to .env file${NC}"
            log "Added Solana keypair to .env file"
        fi
        
        # Airdrop some SOL for development (if on devnet or testnet)
        if [[ "$DEFAULT_ENVIRONMENT" == "devnet" || "$DEFAULT_ENVIRONMENT" == "testnet" ]]; then
            if confirm "Would you like to request an airdrop for development?"; then
                echo "Airdropping SOL to your account for development..."
                log "Requesting SOL airdrop"
                
                PUBKEY=$(solana address)
                
                if [[ "$DEFAULT_ENVIRONMENT" == "devnet" ]]; then
                    solana airdrop 2 "$PUBKEY" || echo -e "${YELLOW}Warning: Airdrop failed. This is normal if you've recently requested an airdrop.${NC}"
                else
                    solana airdrop 1 "$PUBKEY" || echo -e "${YELLOW}Warning: Airdrop failed. This is normal if you've recently requested an airdrop.${NC}"
                fi
                
                echo -e "${GREEN}✓ Your Solana address: ${PUBKEY}${NC}"
                echo -e "${GREEN}✓ Current balance: $(solana balance) SOL${NC}"
                log "Current balance: $(solana balance) SOL"
            fi
        fi
    fi
    
    echo -e "${GREEN}Environment configured!${NC}\n"
    log "Environment configured"
}

# Install project dependencies
install_dependencies() {
    echo -e "${BLUE}${BOLD}Installing project dependencies...${NC}"
    log "Installing project dependencies"
    
    # Install root project dependencies
    echo "Installing root project dependencies..."
    log "Installing root project dependencies"
    npm install
    echo -e "${GREEN}✓ Root dependencies installed${NC}"
    log "Root dependencies installed"
    
    # Initialize and update git submodules if any
    if [ -f .gitmodules ]; then
        echo "Initializing git submodules..."
        log "Initializing git submodules"
        git submodule update --init --recursive
        echo -e "${GREEN}✓ Git submodules initialized${NC}"
        log "Git submodules initialized"
    fi
    
    # Install dependencies for each package
    echo "Installing package dependencies..."
    log "Installing package dependencies"
    
    # Check if lerna is available in node_modules
    if [ -f "node_modules/.bin/lerna" ]; then
        echo "Using Lerna to bootstrap packages..."
        log "Using Lerna to bootstrap packages"
        npx lerna bootstrap
        echo -e "${GREEN}✓ All packages bootstrapped with Lerna${NC}"
        log "All packages bootstrapped with Lerna"
    else
        # Manually install dependencies for each package
        for pkg in packages/*/; do
            if [ -f "${pkg}package.json" ]; then
                echo "Installing dependencies for ${pkg}..."
                log "Installing dependencies for ${pkg}"