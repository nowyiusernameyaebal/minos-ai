/**
 * @file Example demonstrating how to create a vault using the Minos-AI SDK
 * @module @minos-ai/sdk/examples/create-vault
 * @description This example shows how to create a vault with various configuration options
 */

import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { BN } from '@project-serum/anchor';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { 
  createMinosClient, 
  AssetType, 
  RiskLevel, 
  TimeHorizon, 
  AiModelType,
  VaultStatus
} from '../src';

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '../.env') });

/**
 * Main function to demonstrate vault creation
 */
async function main() {
  try {
    console.log('Minos-AI SDK - Create Vault Example');
    console.log('-----------------------------------');

    // ===== Setup connection and wallet =====
    console.log('\n1. Setting up connection and wallet...');
    
    // Check if required environment variables are set
    const requiredEnvVars = ['SOLANA_RPC_URL', 'WALLET_PRIVATE_KEY'];
    for (const envVar of requiredEnvVars) {
      if (!process.env[envVar]) {
        throw new Error(`Missing required environment variable: ${envVar}`);
      }
    }

    // Setup Solana connection
    const rpcUrl = process.env.SOLANA_RPC_URL!;
    const connection = new Connection(rpcUrl, 'confirmed');
    console.log(`Connecting to Solana network at ${rpcUrl}`);
    
    // Get network name
    const genesisHash = await connection.getGenesisHash();
    const networkName = getNetworkName(genesisHash);
    console.log(`Connected to Solana ${networkName} network`);

    // Setup wallet
    let secretKey: Uint8Array;
    if (process.env.WALLET_PRIVATE_KEY!.includes('[')) {
      // Handle array format
      secretKey = Uint8Array.from(JSON.parse(process.env.WALLET_PRIVATE_KEY!));
    } else {
      // Handle base58 format
      secretKey = Uint8Array.from(Buffer.from(process.env.WALLET_PRIVATE_KEY!, 'base64'));
    }
    
    const keypair = Keypair.fromSecretKey(secretKey);
    console.log(`Using wallet address: ${keypair.publicKey.toString()}`);

    // Get wallet balance
    const balance = await connection.getBalance(keypair.publicKey);
    console.log(`Wallet balance: ${balance / 1e9} SOL`);

    if (balance < 10000000) {
      throw new Error('Insufficient balance to create a vault. Minimum required: 0.01 SOL');
    }

    // ===== Initialize Minos-AI client =====
    console.log('\n2. Initializing Minos-AI client...');
    const client = createMinosClient(connection, keypair, {
      // Determine environment based on network
      environment: networkName === 'mainnet-beta' 
        ? 'mainnet-beta' 
        : (networkName === 'devnet' ? 'devnet' : 'localnet'),
    });
    console.log('Client initialized successfully');

    // ===== Get vault creation parameters from command line args =====
    console.log('\n3. Preparing vault creation parameters...');
    
    // Parse command line arguments or use defaults
    const args = parseCommandLineArgs();
    
    const vaultName = args.name || 'My First AI Vault';
    const assetType = getAssetType(args.assetType || 'SOL');
    const tokenMint = args.tokenMint ? new PublicKey(args.tokenMint) : undefined;
    const riskLevel = getRiskLevel(args.riskLevel || 'MODERATE');
    const timeHorizon = getTimeHorizon(args.timeHorizon || 'MEDIUM_TERM');
    const aiModelType = getAiModelType(args.aiModelType || 'ARIADNE');
    const initialDeposit = new BN(args.initialDeposit || 100000000); // 0.1 SOL by default

    console.log(`Vault name: ${vaultName}`);
    console.log(`Asset type: ${assetType}`);
    if (tokenMint) {
      console.log(`Token mint: ${tokenMint.toString()}`);
    }
    console.log(`Risk level: ${riskLevel}`);
    console.log(`Time horizon: ${timeHorizon}`);
    console.log(`AI model type: ${aiModelType}`);
    console.log(`Initial deposit: ${initialDeposit.toString()} (${initialDeposit.toNumber() / 1e9} SOL)`);

    // Get strategy parameters based on AI model type
    const strategyParams = getStrategyParams(aiModelType);
    
    console.log('Strategy parameters:');
    console.log(JSON.stringify(strategyParams, null, 2));

    // ===== Create the vault =====
    console.log('\n4. Creating vault...');
    console.log('This may take a few moments...');
    
    const vault = await client.vault.createVault({
      name: vaultName,
      assetType,
      tokenMint,
      riskLevel,
      timeHorizon,
      aiModelType,
      initialDeposit,
      strategyParams,
      // Additional optional parameters
      description: 'Created using the Minos-AI SDK example',
      fees: {
        managementFee: 2.0, // 2.0% annual management fee
        performanceFee: 20.0, // 20.0% performance fee
        withdrawalFee: 0.5, // 0.5% withdrawal fee
      },
      minDeposit: new BN(10000000), // 0.01 SOL minimum deposit
      maxCapacity: new BN(1000000000000), // 1000 SOL maximum capacity
      allowWithdrawals: true,
      isPublic: true,
    });

    console.log('\n✓ Vault created successfully!');
    console.log('\nVault details:');
    console.log(`ID: ${vault.id.toString()}`);
    console.log(`Name: ${vault.name}`);
    console.log(`Authority: ${vault.authority.toString()}`);
    console.log(`Asset type: ${vault.assetType}`);
    console.log(`Status: ${vault.status}`);
    console.log(`AUM: ${vault.aum.toString()} (${vault.aum.toNumber() / 1e9} SOL)`);
    console.log(`Created at: ${new Date(vault.createdAt.toNumber() * 1000).toLocaleString()}`);
    
    // ===== Create an AI agent for the vault =====
    console.log('\n5. Creating AI agent for the vault...');
    
    const agent = await client.agent.createAgent({
      name: `${vaultName} Agent`,
      description: 'AI trading agent for the vault',
      modelType: aiModelType,
      riskLevel,
      timeHorizon,
      vaultId: vault.id,
      strategyParams,
    });

    console.log('\n✓ Agent created successfully!');
    console.log('\nAgent details:');
    console.log(`ID: ${agent.id.toString()}`);
    console.log(`Name: ${agent.name}`);
    console.log(`Model type: ${agent.modelType}`);
    console.log(`Risk level: ${agent.riskLevel}`);
    console.log(`Vault ID: ${agent.vaultId?.toString()}`);
    console.log(`Active: ${agent.isActive}`);
    
    // ===== Display vault and agent links =====
    console.log('\nLinks to view your vault and agent:');
    
    const baseUrl = getExplorerUrl(networkName);
    console.log(`Vault: ${baseUrl}/address/${vault.id.toString()}`);
    console.log(`Agent: ${baseUrl}/address/${agent.id.toString()}`);
    
    // ===== Next steps =====
    console.log('\nNext steps:');
    console.log('1. Visit the Minos-AI dashboard to monitor your vault performance');
    console.log('2. Use client.vault.deposit() to add more funds to your vault');
    console.log('3. Use client.vault.withdraw() to withdraw funds from your vault');
    console.log('4. Use client.agent.updateStrategy() to adjust the AI strategy parameters');
    console.log('\nThank you for using Minos-AI SDK!');

  } catch (error) {
    console.error('\n❌ Error:');
    if (error instanceof Error) {
      console.error(`- Message: ${error.message}`);
      console.error(`- Stack: ${error.stack}`);
    } else {
      console.error(error);
    }
    process.exit(1);
  }
}

/**
 * Gets the strategy parameters based on the AI model type
 * 
 * @param {AiModelType} modelType - AI model type
 * @returns {Object} Strategy parameters
 */
function getStrategyParams(modelType: AiModelType): any {
  switch (modelType) {
    case AiModelType.ARIADNE:
      return {
        timeInterval: '1h',
        lookbackPeriod: 14,
        indicators: ['RSI', 'MACD', 'BB'],
        riskAllocation: 5,
        maxDrawdown: 10,
        takeProfit: 15,
        stopLoss: 5,
        tradeSizingMethod: 'percentage',
        positionSizePercentage: 10
      };
    
    case AiModelType.ANDROGEUS:
      return {
        technicalAnalysis: {
          maPeriods: [9, 21, 50],
          rsi: {
            period: 14,
            overbought: 70,
            oversold: 30
          },
          macd: {
            fastPeriod: 12,
            slowPeriod: 26,
            signalPeriod: 9
          },
          bollingerBands: {
            period: 20,
            deviations: 2
          }
        },
        riskManagement: {
          maxPositionSize: 20,
          maxRiskPerTrade: 2,
          stopLossType: 'atr',
          stopLossParam: 2,
          takeProfitType: 'risk-reward',
          takeProfitParam: 3
        },
        positionManagement: {
          partialExits: true,
          scaleIn: false,
          positionSizingMethod: 'percentage'
        }
      };
    
    case AiModelType.DEUCALION:
      return {
        dataSources: ['twitter', 'reddit', 'news'],
        sentiment: {
          minSentimentScore: 0.6,
          lookbackPeriod: 24,
          useAdvancedNlp: true
        },
        marketData: {
          correlateWithMarket: true,
          sources: ['binance', 'coinmarketcap']
        },
        copyTrading: {
          enabled: true,
          minTraderScore: 75,
          maxTradersToFollow: 5,
          allocationWeightMethod: 'performance'
        },
        riskManagement: {
          maxAllocationPerAsset: 20,
          maxAllocationPerTrader: 10,
          positionSizingMethod: 'proportional'
        }
      };
    
    case AiModelType.CUSTOM:
      return {
        strategyDefinition: {
          name: 'Custom Strategy',
          version: '1.0.0',
          description: 'A custom trading strategy'
        },
        parameters: {
          param1: 'value1',
          param2: 'value2',
          param3: 123
        },
        backtestingResults: {
          sharpeRatio: 1.5,
          maxDrawdown: 15,
          totalReturn: 45,
          winRate: 65
        },
        riskManagement: {
          maxLoss: 10,
          positionSizing: 'dynamic'
        }
      };
    
    default:
      return {};
  }
}

/**
 * Parse command line arguments
 * 
 * @returns {Object} Parsed arguments
 */
function parseCommandLineArgs(): Record<string, any> {
  const args: Record<string, any> = {};
  
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    
    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      const value = process.argv[i + 1];
      
      if (value && !value.startsWith('--')) {
        args[key] = value;
        i++;
      } else {
        args[key] = true;
      }
    }
  }
  
  return args;
}

/**
 * Gets the network name from the genesis hash
 * 
 * @param {string} genesisHash - Genesis hash
 * @returns {string} Network name
 */
function getNetworkName(genesisHash: string): string {
  // Mainnet genesis hash
  if (genesisHash === '5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d') {
    return 'mainnet-beta';
  }
  
  // Devnet genesis hash
  if (genesisHash === 'EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG') {
    return 'devnet';
  }
  
  // Testnet genesis hash
  if (genesisHash === '4uhcVJyU9pJkvQyS88uRDiswHXSCkY3zQawwpjk2NsNY') {
    return 'testnet';
  }
  
  // Default to localnet
  return 'localnet';
}

/**
 * Gets the explorer URL for the network
 * 
 * @param {string} network - Network name
 * @returns {string} Explorer URL
 */
function getExplorerUrl(network: string): string {
  switch (network) {
    case 'mainnet-beta':
      return 'https://explorer.solana.com';
    case 'devnet':
      return 'https://explorer.solana.com?cluster=devnet';
    case 'testnet':
      return 'https://explorer.solana.com?cluster=testnet';
    default:
      return 'https://explorer.solana.com?cluster=custom';
  }
}

/**
 * Gets the asset type from a string
 * 
 * @param {string} assetTypeStr - Asset type string
 * @returns {AssetType} Asset type enum
 */
function getAssetType(assetTypeStr: string): AssetType {
  switch (assetTypeStr.toUpperCase()) {
    case 'SOL':
      return AssetType.SOL;
    case 'SPL_TOKEN':
    case 'SPL':
      return AssetType.SPL_TOKEN;
    case 'USDC':
      return AssetType.USDC;
    case 'BTC':
      return AssetType.BTC;
    case 'ETH':
      return AssetType.ETH;
    default:
      return AssetType.SOL;
  }
}

/**
 * Gets the risk level from a string
 * 
 * @param {string} riskLevelStr - Risk level string
 * @returns {RiskLevel} Risk level enum
 */
function getRiskLevel(riskLevelStr: string): RiskLevel {
  switch (riskLevelStr.toUpperCase()) {
    case 'CONSERVATIVE':
      return RiskLevel.CONSERVATIVE;
    case 'MODERATE':
      return RiskLevel.MODERATE;
    case 'AGGRESSIVE':
      return RiskLevel.AGGRESSIVE;
    case 'CUSTOM':
      return RiskLevel.CUSTOM;
    default:
      return RiskLevel.MODERATE;
  }
}

/**
 * Gets the time horizon from a string
 * 
 * @param {string} timeHorizonStr - Time horizon string
 * @returns {TimeHorizon} Time horizon enum
 */
function getTimeHorizon(timeHorizonStr: string): TimeHorizon {
  switch (timeHorizonStr.toUpperCase()) {
    case 'SHORT_TERM':
    case 'SHORT':
      return TimeHorizon.SHORT_TERM;
    case 'MEDIUM_TERM':
    case 'MEDIUM':
      return TimeHorizon.MEDIUM_TERM;
    case 'LONG_TERM':
    case 'LONG':
      return TimeHorizon.LONG_TERM;
    default:
      return TimeHorizon.MEDIUM_TERM;
  }
}

/**
 * Gets the AI model type from a string
 * 
 * @param {string} modelTypeStr - AI model type string
 * @returns {AiModelType} AI model type enum
 */
function getAiModelType(modelTypeStr: string): AiModelType {
  switch (modelTypeStr.toUpperCase()) {
    case 'ARIADNE':
      return AiModelType.ARIADNE;
    case 'ANDROGEUS':
      return AiModelType.ANDROGEUS;
    case 'DEUCALION':
      return AiModelType.DEUCALION;
    case 'CUSTOM':
      return AiModelType.CUSTOM;
    default:
      return AiModelType.ARIADNE;
  }
}

// Run the main function
main().catch(console.error);