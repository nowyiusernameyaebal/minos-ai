/**
 * @file Main entry point for the Minos-AI SDK
 * @module @minos-ai/sdk
 * @description A TypeScript SDK for interacting with the Minos-AI DeFi platform on Solana
 */

import { Connection, Keypair, PublicKey } from '@solana/web3.js';
import { AnchorProvider, Program, Wallet } from '@project-serum/anchor';

// Export client class
import { MinosClient } from './client';
export { MinosClient };

// Export module components
import * as VaultModule from './vault';
import * as AgentModule from './agent';

// Export utility types
import * as Types from './types';

// Export constants
import {
  DEFAULT_PROGRAM_IDS,
  DEFAULT_API_ENDPOINT,
  DEFAULT_COMMITMENT,
  DEFAULT_TIMEOUT,
  DEFAULT_MAX_RETRIES,
  SDK_VERSION,
} from './constants';

/**
 * Create a new Minos client instance with the provided configuration
 * 
 * @param {Connection} connection - Solana connection instance
 * @param {Wallet | Keypair} wallet - Wallet or keypair for signing transactions
 * @param {Partial<Types.MinosClientConfig>} config - Additional configuration options
 * @returns {MinosClient} A new Minos client instance
 * 
 * @example
 * ```typescript
 * import { Connection, Keypair } from '@solana/web3.js';
 * import { createMinosClient } from '@minos-ai/sdk';
 * 
 * // Create a Solana connection
 * const connection = new Connection('https://api.mainnet-beta.solana.com');
 * 
 * // Create or load a keypair
 * const keypair = Keypair.fromSecretKey(Buffer.from(process.env.PRIVATE_KEY, 'base64'));
 * 
 * // Create the Minos client
 * const client = createMinosClient(connection, keypair, {
 *   environment: 'mainnet-beta',
 *   apiEndpoint: 'https://api.minos-ai.io',
 * });
 * ```
 */
export function createMinosClient(
  connection: Connection,
  wallet: Wallet | Keypair,
  config: Partial<Types.MinosClientConfig> = {},
): MinosClient {
  // Convert Keypair to AnchorWallet if needed
  const anchorWallet = 'publicKey' in wallet 
    ? wallet 
    : {
        publicKey: wallet.publicKey,
        signTransaction: async (tx) => {
          tx.partialSign(wallet);
          return tx;
        },
        signAllTransactions: async (txs) => {
          txs.forEach((tx) => tx.partialSign(wallet));
          return txs;
        },
      };

  // Create AnchorProvider if not provided
  const provider = config.provider || 
    new AnchorProvider(
      connection,
      anchorWallet,
      { commitment: config.commitment || DEFAULT_COMMITMENT }
    );

  // Return new client instance
  return new MinosClient({
    connection,
    provider,
    ...config,
  });
}

/**
 * Returns the SDK version
 * @returns {string} The current SDK version
 */
export function version(): string {
  return SDK_VERSION;
}

/**
 * Validate a Solana public key
 * 
 * @param {string} key - The public key to validate
 * @returns {boolean} Whether the public key is valid
 */
export function isValidPublicKey(key: string): boolean {
  try {
    new PublicKey(key);
    return true;
  } catch (error) {
    return false;
  }
}

/**
 * Create a new keypair
 * 
 * @returns {Keypair} A new Solana keypair
 */
export function createKeypair(): Keypair {
  return Keypair.generate();
}

/**
 * Validate a Minos-AI vault address
 * 
 * @param {string} address - The vault address to validate
 * @returns {boolean} Whether the address is a valid vault
 * 
 * @example
 * ```typescript
 * import { isValidVaultAddress } from '@minos-ai/sdk';
 * 
 * const isValid = isValidVaultAddress('7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU');
 * console.log(isValid); // true or false
 * ```
 */
export function isValidVaultAddress(address: string): boolean {
  if (!isValidPublicKey(address)) {
    return false;
  }
  
  // Additional validation logic could be added here
  // For example, checking if the address belongs to the vault program
  
  return true;
}

/**
 * Get the estimated gas fee for a transaction
 * 
 * @param {Connection} connection - Solana connection instance
 * @returns {Promise<number>} The estimated fee in lamports
 */
export async function getEstimatedFee(connection: Connection): Promise<number> {
  const { feeCalculator } = await connection.getRecentBlockhash();
  return feeCalculator.lamportsPerSignature;
}

// Export all submodules
export const Vault = VaultModule;
export const Agent = AgentModule;

// Export types
export {
  Types,
  DEFAULT_PROGRAM_IDS,
  DEFAULT_API_ENDPOINT,
  DEFAULT_COMMITMENT,
  DEFAULT_TIMEOUT,
  DEFAULT_MAX_RETRIES,
};

// Set up and export individual types for convenience
export type {
  // Client types
  Types.MinosClientConfig,
  Types.ProgramIds,
  Types.DefaultProgramIds,
  
  // Vault types
  Types.VaultAccount,
  Types.CreateVaultParams,
  Types.DepositParams,
  Types.WithdrawParams,
  Types.VaultQueryParams,
  Types.VaultPerformance,
  
  // Agent types
  Types.AgentAccount,
  Types.CreateAgentParams,
  Types.AgentQueryParams,
  Types.AgentPerformance,
  
  // Order types
  Types.OrderAccount,
  Types.CreateOrderParams,
  Types.OrderQueryParams,
  
  // Strategy types
  Types.AriadneStrategyParams,
  Types.AndrogeusStrategyParams,
  Types.DeucalionStrategyParams,
  Types.CustomStrategyParams,
  Types.StrategyParams,
  
  // Common types
  Types.TransactionResult,
  Types.ApiResponse,
  Types.PaginatedResponse,
  Types.AccountWithData,
  Types.TransactionOptions,
  Types.ApiRequestOptions,
  Types.MarketData,
  Types.Candlestick,
  Types.Trade,
  Types.DashboardData,
  Types.UserAccount,
};

// Export enums
export {
  Types.NetworkEnvironment,
  Types.AssetType,
  Types.RiskLevel,
  Types.TimeHorizon,
  Types.OrderType,
  Types.OrderSide,
  Types.OrderStatus,
  Types.VaultStatus,
  Types.AiModelType,
  Types.EventType,
  Types.ErrorCode,
  Types.WebhookEventType,
};

// Export namespaces
export {
  Types.VaultIDL,
  Types.AgentIDL,
  Types.GovernanceIDL,
};

// Export error class
export {
  Types.MinosSdkError,
};

// Default export
export default {
  createMinosClient,
  version,
  isValidPublicKey,
  createKeypair,
  isValidVaultAddress,
  getEstimatedFee,
  MinosClient,
  Vault: VaultModule,
  Agent: AgentModule,
  Types,
};