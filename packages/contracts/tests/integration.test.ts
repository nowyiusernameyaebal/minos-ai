/**
 * @file packages/contracts/tests/integration.test.ts
 * @description Integration tests for Vault and AI Agent contracts working together
 * @version 1.0.0
 */

import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { Vault } from "../target/types/vault";
import { AiAgent } from "../target/types/ai_agent";
import {
  PublicKey,
  Keypair,
  LAMPORTS_PER_SOL,
  SystemProgram,
  Transaction,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import {
  TOKEN_PROGRAM_ID,
  ASSOCIATED_TOKEN_PROGRAM_ID,
  getAssociatedTokenAddress,
  createAssociatedTokenAccountInstruction,
  createMintToInstruction,
  createInitializeMintInstruction,
  MINT_SIZE,
  getMinimumBalanceForRentExemptMint,
} from "@solana/spl-token";
import { expect } from "chai";
import { BN } from "bn.js";

describe("Vault-AI Agent Integration Tests", () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const vaultProgram = anchor.workspace.Vault as Program<Vault>;
  const agentProgram = anchor.workspace.AiAgent as Program<AiAgent>;
  const connection = provider.connection;
  const wallet = provider.wallet as anchor.Wallet;

  // Test accounts and PDAs
  let vaultPda: PublicKey;
  let vaultBump: number;
  let agentPda: PublicKey;
  let agentBump: number;
  let modelPda: PublicKey;
  let modelBump: number;
  let tradePda: PublicKey;
  let tradeBump: number;
  let userSharesPda: PublicKey;
  let userSharesBump: number;

  // Token accounts
  let vaultTokenAccount: PublicKey;
  let agentTokenAccount: PublicKey;
  let userTokenAccount: PublicKey;
  let managerTokenAccount: PublicKey;
  let treasuryTokenAccount: PublicKey;

  // Keypairs
  let mintKeypair: Keypair;
  let userKeypair: Keypair;
  let managerKeypair: Keypair;
  let treasuryKeypair: Keypair;
  let agentOwnerKeypair: Keypair;

  // Test constants
  const INITIAL_SUPPLY = new BN(10_000_000 * LAMPORTS_PER_SOL);
  const VAULT_DEPOSIT = new BN(1_000 * LAMPORTS_PER_SOL);
  const MANAGEMENT_FEE = 200; // 2%
  const PERFORMANCE_FEE = 1000; // 10%
  const MIN_DEPOSIT = new BN(10 * LAMPORTS_PER_SOL);
  const MAX_DEPOSIT = new BN(10_000 * LAMPORTS_PER_SOL);
  const AI_CONFIDENCE_THRESHOLD = 75;
  const MAX_TRADE_SIZE = new BN(500 * LAMPORTS_PER_SOL);
  const MIN_TRADE_SIZE = new BN(5 * LAMPORTS_PER_SOL);
  const RISK_TOLERANCE = 60;

  before(async () => {
    // Initialize keypairs
    userKeypair = Keypair.generate();
    managerKeypair = Keypair.generate();
    treasuryKeypair = Keypair.generate();
    agentOwnerKeypair = Keypair.generate();
    mintKeypair = Keypair.generate();

    // Airdrop SOL to all test accounts
    const accounts = [
      userKeypair.publicKey,
      managerKeypair.publicKey,
      treasuryKeypair.publicKey,
      agentOwnerKeypair.publicKey,
      wallet.publicKey,
    ];

    for (const account of accounts) {
      await connection.requestAirdrop(account, 10 * LAMPORTS_PER_SOL);
    }

    // Wait for airdrop confirmations
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Create mint
    const mintRent = await getMinimumBalanceForRentExemptMint(connection);
    const createMintTx = new Transaction().add(
      SystemProgram.createAccount({
        fromPubkey: wallet.publicKey,
        newAccountPubkey: mintKeypair.publicKey,
        space: MINT_SIZE,
        lamports: mintRent,
        programId: TOKEN_PROGRAM_ID,
      }),
      createInitializeMintInstruction(
        mintKeypair.publicKey,
        6, // decimals
        wallet.publicKey,
        wallet.publicKey
      )
    );

    await sendAndConfirmTransaction(connection, createMintTx, [wallet.payer, mintKeypair]);

    // Derive PDAs
    [vaultPda, vaultBump] = await PublicKey.findProgramAddress(
      [Buffer.from("vault"), mintKeypair.publicKey.toBuffer()],
      vaultProgram.programId
    );

    [agentPda, agentBump] = await PublicKey.findProgramAddress(
      [Buffer.from("agent"), agentOwnerKeypair.publicKey.toBuffer()],
      agentProgram.programId
    );

    [modelPda, modelBump] = await PublicKey.findProgramAddress(
      [Buffer.from("model"), agentPda.toBuffer()],
      agentProgram.programId
    );

    [userSharesPda, userSharesBump] = await PublicKey.findProgramAddress(
      [Buffer.from("shares"), vaultPda.toBuffer(), userKeypair.publicKey.toBuffer()],
      vaultProgram.programId
    );

    // Create associated token accounts
    vaultTokenAccount = await getAssociatedTokenAddress(
      mintKeypair.publicKey,
      vaultPda,
      true // allowOwnerOffCurve
    );

    agentTokenAccount = await getAssociatedTokenAddress(
      mintKeypair.publicKey,
      agentPda,
      true // allowOwnerOffCurve
    );

    userTokenAccount = await getAssociatedTokenAddress(
      mintKeypair.publicKey,
      userKeypair.publicKey
    );

    managerTokenAccount = await getAssociatedTokenAddress(
      mintKeypair.publicKey,
      managerKeypair.publicKey
    );

    treasuryTokenAccount = await getAssociatedTokenAddress(
      mintKeypair.publicKey,
      treasuryKeypair.publicKey
    );

    // Create user token account and mint initial supply
    const createUserTokenAccountTx = new Transaction().add(
      createAssociatedTokenAccountInstruction(
        wallet.publicKey,
        userTokenAccount,
        userKeypair.publicKey,
        mintKeypair.publicKey
      )
    );

    await sendAndConfirmTransaction(connection, createUserTokenAccountTx, [wallet.payer]);

    // Mint tokens to user
    const mintToTx = new Transaction().add(
      createMintToInstruction(
        mintKeypair.publicKey,
        userTokenAccount,
        wallet.publicKey,
        INITIAL_SUPPLY.toNumber()
      )
    );

    await sendAndConfirmTransaction(connection, mintToTx, [wallet.payer]);
  });

  describe("Initial Setup", () => {
    it("should initialize vault successfully", async () => {
      const tx = await vaultProgram.methods
        .initializeVault(
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          MIN_DEPOSIT,
          MAX_DEPOSIT
        )
        .accounts({
          vault: vaultPda,
          manager: managerKeypair.publicKey,
          treasury: treasuryKeypair.publicKey,
          acceptedMint: mintKeypair.publicKey,
          vaultTokenAccount: vaultTokenAccount,
          payer: wallet.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
          rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        })
        .signers([wallet.payer])
        .rpc();

      console.log("Vault initialization:", tx);

      const vaultAccount = await vaultProgram.account.vault.fetch(vaultPda);
      expect(vaultAccount.isActive).to.be.true;
      expect(vaultAccount.totalAssets.toString()).to.equal("0");
    });

    it("should initialize AI agent for the vault", async () => {
      const tx = await agentProgram.methods
        .initializeAgent(
          "Vault AI Trading Agent",
          0, // Conservative strategy
          AI_CONFIDENCE_THRESHOLD,
          MAX_TRADE_SIZE,
          MIN_TRADE_SIZE,
          RISK_TOLERANCE,
          500, // 5% rebalance threshold
          1000, // 10% stop loss
          1500 // 15% take profit
        )
        .accounts({
          agent: agentPda,
          model: modelPda,
          owner: agentOwnerKeypair.publicKey,
          vault: vaultPda,
          agentTokenAccount: agentTokenAccount,
          payer: wallet.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
          rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        })
        .signers([wallet.payer])
        .rpc();

      console.log("AI Agent initialization:", tx);

      const agentAccount = await agentProgram.account.aiAgent.fetch(agentPda);
      expect(agentAccount.isActive).to.be.true;
      expect(agentAccount.vault.toString()).to.equal(vaultPda.toString());
    });

    it("should set the AI agent as a strategy for the vault", async () => {
      const tx = await vaultProgram.methods
        .addStrategy(agentPda, 7500) // 75% allocation to AI agent
        .accounts({
          vault: vaultPda,
          manager: managerKeypair.publicKey,
        })
        .signers([managerKeypair])
        .rpc();

      console.log("Add AI agent as strategy:", tx);

      const vaultAccount = await vaultProgram.account.vault.fetch(vaultPda);
      expect(vaultAccount.strategies).to.have.length(1);
      expect(vaultAccount.strategies[0].strategyAccount.toString()).to.equal(agentPda.toString());
      expect(vaultAccount.strategies[0].allocation).to.equal(7500);
    });
  });

  describe("User Deposits and AI Trading", () => {
    it("should allow user to deposit into vault", async () => {
      const tx = await vaultProgram.methods
        .deposit(VAULT_DEPOSIT)
        .accounts({
          vault: vaultPda,
          userShares: userSharesPda,
          userTokenAccount: userTokenAccount,
          vaultTokenAccount: vaultTokenAccount,
          user: userKeypair.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
          rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        })
        .signers([userKeypair])
        .rpc();

      console.log("User deposit:", tx);

      const vaultAccount = await vaultProgram.account.vault.fetch(vaultPda);
      expect(vaultAccount.totalAssets.toString()).to.equal(VAULT_DEPOSIT.toString());

      const userShares = await vaultProgram.account.userShares.fetch(userSharesPda);
      expect(userShares.shares.toString()).to.equal(VAULT_DEPOSIT.toString());
    });

    it("should train AI model with historical data", async () => {
      // Simulate training data
      const trainingData = Array(20).fill(0).map((_, i) => ({
        features: [
          100 + i * 5, // Price trend
          50 + Math.random() * 50, // RSI
          Math.random() * 1000, // Volume
        ],
        label: i % 2, // Binary classification (buy/sell)
      }));

      const tx = await agentProgram.methods
        .startTraining(trainingData)
        .accounts({
          agent: agentPda,
          model: modelPda,
          owner: agentOwnerKeypair.publicKey,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      console.log("Start AI training:", tx);

      // Update model weights (simulating training completion)
      const weights = [
        [0.2, 0.3, 0.1],
        [0.4, 0.1, 0.5],
        [0.3, 0.6, 0.4]
      ];
      const biases = [0.1, 0.2, 0.3];

      await agentProgram.methods
        .updateModelWeights(weights, biases, 85)
        .accounts({
          agent: agentPda,
          model: modelPda,
          owner: agentOwnerKeypair.publicKey,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      // Complete training
      await agentProgram.methods
        .completeTraining()
        .accounts({
          agent: agentPda,
          model: modelPda,
          owner: agentOwnerKeypair.publicKey,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      const modelAccount = await agentProgram.account.aiModel.fetch(modelPda);
      expect(modelAccount.isTraining).to.be.false;
      expect(modelAccount.accuracy).to.equal(85);
    });

    it("should generate prediction and execute trades", async () => {
      // Generate market data for prediction
      const marketData = {
        currentPrice: new BN(100 * LAMPORTS_PER_SOL),
        volume24h: new BN(1_000_000 * LAMPORTS_PER_SOL),
        volatility: 300, // 3%
        trend: 1, // Bullish
        technicalIndicators: [
          { name: "RSI", value: 45 },
          { name: "MACD", value: 150 },
          { name: "EMA", value: 98 * LAMPORTS_PER_SOL },
        ],
      };

      // Generate prediction
      await agentProgram.methods
        .generatePrediction(marketData)
        .accounts({
          agent: agentPda,
          model: modelPda,
          owner: agentOwnerKeypair.publicKey,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      // Execute trade based on prediction
      [tradePda, tradeBump] = await PublicKey.findProgramAddress(
        [Buffer.from("trade"), agentPda.toBuffer(), new BN(0).toArrayLike(Buffer, "le", 8)],
        agentProgram.programId
      );

      const tradeAmount = new BN(100 * LAMPORTS_PER_SOL);
      const expectedPrice = new BN(102 * LAMPORTS_PER_SOL);

      const tradeTx = await agentProgram.methods
        .executeTrade(
          0, // Buy
          tradeAmount,
          expectedPrice,
          88 // High confidence
        )
        .accounts({
          agent: agentPda,
          trade: tradePda,
          vault: vaultPda,
          agentTokenAccount: agentTokenAccount,
          vaultTokenAccount: vaultTokenAccount,
          owner: agentOwnerKeypair.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
          rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      console.log("Execute trade:", tradeTx);

      const tradeAccount = await agentProgram.account.tradeRecord.fetch(tradePda);
      expect(tradeAccount.isExecuted).to.be.true;
      expect(tradeAccount.confidence).to.equal(88);

      const agentAccount = await agentProgram.account.aiAgent.fetch(agentPda);
      expect(agentAccount.totalTrades.toString()).to.equal("1");
    });
  });

  describe("Profit Distribution", () => {
    it("should simulate profit and distribute performance fees", async () => {
      // Simulate profit by adding tokens to vault (representing trading gains)
      const simulatedProfit = new BN(200 * LAMPORTS_PER_SOL);

      // Create manager and treasury token accounts
      const createManagerTokenAccountTx = new Transaction().add(
        createAssociatedTokenAccountInstruction(
          wallet.publicKey,
          managerTokenAccount,
          managerKeypair.publicKey,
          mintKeypair.publicKey
        )
      );

      const createTreasuryTokenAccountTx = new Transaction().add(
        createAssociatedTokenAccountInstruction(
          wallet.publicKey,
          treasuryTokenAccount,
          treasuryKeypair.publicKey,
          mintKeypair.publicKey
        )
      );

      await sendAndConfirmTransaction(connection, createManagerTokenAccountTx, [wallet.payer]);
      await sendAndConfirmTransaction(connection, createTreasuryTokenAccountTx, [wallet.payer]);

      // Mint profit tokens directly to vault to simulate trading gains
      const mintProfitTx = new Transaction().add(
        createMintToInstruction(
          mintKeypair.publicKey,
          vaultTokenAccount,
          wallet.publicKey,
          simulatedProfit.toNumber()
        )
      );

      await sendAndConfirmTransaction(connection, mintProfitTx, [wallet.payer]);

      // Collect performance fees
      const feeCollectionTx = await vaultProgram.methods
        .collectPerformanceFee(simulatedProfit)
        .accounts({
          vault: vaultPda,
          vaultTokenAccount: vaultTokenAccount,
          managerTokenAccount: managerTokenAccount,
          treasuryTokenAccount: treasuryTokenAccount,
          manager: managerKeypair.publicKey,
          treasury: treasuryKeypair.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
        })
        .signers([managerKeypair])
        .rpc();

      console.log("Collect performance fees:", feeCollectionTx);

      // Verify fee distribution
      const managerBalance = await connection.getTokenAccountBalance(managerTokenAccount);
      const treasuryBalance = await connection.getTokenAccountBalance(treasuryTokenAccount);

      const expectedManagerFee = simulatedProfit.mul(new BN(PERFORMANCE_FEE)).div(new BN(10000)).div(new BN(2));
      const expectedTreasuryFee = expectedManagerFee;

      expect(managerBalance.value.amount).to.equal(expectedManagerFee.toString());
      expect(treasuryBalance.value.amount).to.equal(expectedTreasuryFee.toString());
    });
  });

  describe("Risk Management Integration", () => {
    it("should trigger portfolio rebalancing", async () => {
      const rebalanceThreshold = 500; // 5%
      
      // Check current allocation vs target
      const vaultAccount = await vaultProgram.account.vault.fetch(vaultPda);
      const totalAssets = vaultAccount.totalAssets;

      // Simulate allocation drift
      const targetAllocations = [
        { mint: mintKeypair.publicKey, targetPercentage: 7500 }, // 75% to AI agent
        { mint: mintKeypair.publicKey, targetPercentage: 2500 }, // 25% cash
      ];

      const rebalanceTx = await agentProgram.methods
        .rebalancePortfolio(targetAllocations)
        .accounts({
          agent: agentPda,
          vault: vaultPda,
          owner: agentOwnerKeypair.publicKey,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      console.log("Rebalance portfolio:", rebalanceTx);

      const agentAccount = await agentProgram.account.aiAgent.fetch(agentPda);
      expect(agentAccount.rebalanceCount.toString()).to.equal("1");
      expect(agentAccount.lastRebalance.gt(new BN(0))).to.be.true;
    });

    it("should execute stop-loss when price drops", async () => {
      const [stopLossTradePda] = await PublicKey.findProgramAddress(
        [Buffer.from("trade"), agentPda.toBuffer(), new BN(1).toArrayLike(Buffer, "le", 8)],
        agentProgram.programId
      );

      // Simulate price drop triggering stop-loss
      const stopLossPrice = new BN(85 * LAMPORTS_PER_SOL); // 15% drop
      const sellAmount = new BN(50 * LAMPORTS_PER_SOL);

      const stopLossTx = await agentProgram.methods
        .executeStopLoss(stopLossPrice, sellAmount)
        .accounts({
          agent: agentPda,
          trade: stopLossTradePda,
          vault: vaultPda,
          agentTokenAccount: agentTokenAccount,
          vaultTokenAccount: vaultTokenAccount,
          owner: agentOwnerKeypair.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
          rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      console.log("Execute stop-loss:", stopLossTx);

      const tradeAccount = await agentProgram.account.tradeRecord.fetch(stopLossTradePda);
      expect(tradeAccount.isStopLoss).to.be.true;
      expect(tradeAccount.tradeType).to.equal(1); // Sell
    });
  });

  describe("User Withdrawals", () => {
    it("should allow user to withdraw with correct share value", async () => {
      const userShares = await vaultProgram.account.userShares.fetch(userSharesPda);
      const withdrawAmount = userShares.shares.div(new BN(2)); // Withdraw 50%

      const preWithdrawBalance = await connection.getTokenAccountBalance(userTokenAccount);

      const withdrawTx = await vaultProgram.methods
        .withdraw(withdrawAmount)
        .accounts({
          vault: vaultPda,
          userShares: userSharesPda,
          userTokenAccount: userTokenAccount,
          vaultTokenAccount: vaultTokenAccount,
          user: userKeypair.publicKey,
          tokenProgram: TOKEN_PROGRAM_ID,
        })
        .signers([userKeypair])
        .rpc();

      console.log("User withdrawal:", withdrawTx);

      const postWithdrawBalance = await connection.getTokenAccountBalance(userTokenAccount);
      const balanceIncrease = new BN(postWithdrawBalance.value.amount).sub(new BN(preWithdrawBalance.value.amount));

      // The withdrawn amount should reflect any profits or losses from AI trading
      expect(balanceIncrease.gt(new BN(0))).to.be.true;
      console.log("Withdrawal amount:", balanceIncrease.toString());
    });
  });

  describe("Emergency Procedures", () => {
    it("should allow vault manager to pause vault in emergency", async () => {
      const pauseTx = await vaultProgram.methods
        .pauseVault()
        .accounts({
          vault: vaultPda,
          manager: managerKeypair.publicKey,
        })
        .signers([managerKeypair])
        .rpc();

      console.log("Pause vault:", pauseTx);

      const vaultAccount = await vaultProgram.account.vault.fetch(vaultPda);
      expect(vaultAccount.isActive).to.be.false;
    });

    it("should pause AI agent when vault is paused", async () => {
      const pauseAgentTx = await agentProgram.methods
        .pauseAgent()
        .accounts({
          agent: agentPda,
          owner: agentOwnerKeypair.publicKey,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      console.log("Pause AI agent:", pauseAgentTx);

      const agentAccount = await agentProgram.account.aiAgent.fetch(agentPda);
      expect(agentAccount.isActive).to.be.false;
    });

    it("should reject new trades when both vault and agent are paused", async () => {
      const [emergencyTradePda] = await PublicKey.findProgramAddress(
        [Buffer.from("trade"), agentPda.toBuffer(), new BN(2).toArrayLike(Buffer, "le", 8)],
        agentProgram.programId
      );

      try {
        await agentProgram.methods
          .executeTrade(
            0,
            new BN(50 * LAMPORTS_PER_SOL),
            new BN(100 * LAMPORTS_PER_SOL),
            80
          )
          .accounts({
            agent: agentPda,
            trade: emergencyTradePda,
            vault: vaultPda,
            agentTokenAccount: agentTokenAccount,
            vaultTokenAccount: vaultTokenAccount,
            owner: agentOwnerKeypair.publicKey,
            tokenProgram: TOKEN_PROGRAM_ID,
            systemProgram: SystemProgram.programId,
            rent: anchor.web3.SYSVAR_RENT_PUBKEY,
          })
          .signers([agentOwnerKeypair])
          .rpc();
        
        expect.fail("Should have failed with agent paused");
      } catch (error: any) {
        expect(error.error.errorCode.code).to.equal("AgentPaused");
      }
    });
  });

  describe("Performance Analytics", () => {
    it("should provide comprehensive performance metrics", async () => {
      // Resume vault for final metrics
      await vaultProgram.methods
        .unpauseVault()
        .accounts({
          vault: vaultPda,
          manager: managerKeypair.publicKey,
        })
        .signers([managerKeypair])
        .rpc();

      await agentProgram.methods
        .unpauseAgent()
        .accounts({
          agent: agentPda,
          owner: agentOwnerKeypair.publicKey,
        })
        .signers([agentOwnerKeypair])
        .rpc();

      // Get final metrics
      const vaultAccount = await vaultProgram.account.vault.fetch(vaultPda);
      const agentAccount = await agentProgram.account.aiAgent.fetch(agentPda);
      const modelAccount = await agentProgram.account.aiModel.fetch(modelPda);

      console.log("\n=== Final Performance Metrics ===");
      console.log("Vault Metrics:");
      console.log("- Total Assets:", vaultAccount.totalAssets.toString());
      console.log("- Total Shares:", vaultAccount.totalShares.toString());
      console.log("- Active Strategies:", vaultAccount.strategies.length);

      console.log("\nAI Agent Metrics:");
      console.log("- Total Trades:", agentAccount.totalTrades.toString());
      console.log("- Successful Trades:", agentAccount.successfulTrades.toString());
      console.log("- Win Rate:", agentAccount.successfulTrades.gt(new BN(0)) 
        ? (agentAccount.successfulTrades.toNumber() / agentAccount.totalTrades.toNumber() * 100).toFixed(2) + "%" 
        : "0%");
      console.log("- Rebalance Count:", agentAccount.rebalanceCount.toString());

      console.log("\nModel Metrics:");
      console.log("- Model Type:", modelAccount.modelType);
      console.log("- Accuracy:", modelAccount.accuracy + "%");
      console.log("- Last Prediction Confidence:", modelAccount.lastPredictionConfidence + "%");

      // Calculate APY (simplified)
      const timeElapsed = 1; // Assume 1 day for test
      const initialValue = VAULT_DEPOSIT;
      const currentValue = vaultAccount.totalAssets;
      
      if (currentValue.gt(initialValue)) {
        const profitPercent = currentValue.sub(initialValue).mul(new BN(100)).div(initialValue);
        console.log("\nPerformance:");
        console.log("- Profit:", profitPercent.toString() + "%");
        console.log("- Approx. APY:", profitPercent.mul(new BN(365)).toString() + "%");
      }

      // Verify integration success
      expect(vaultAccount.isActive).to.be.true;
      expect(agentAccount.isActive).to.be.true;
      expect(agentAccount.totalTrades.gt(new BN(0))).to.be.true;
      expect(modelAccount.accuracy).to.be.at.least(75);
    });
  });
});