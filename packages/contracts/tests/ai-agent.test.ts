import * as anchor from '@coral-xyz/anchor';
import { Program, BN, Wallet } from '@coral-xyz/anchor';
import { PublicKey, Keypair, SystemProgram, SYSVAR_RENT_PUBKEY } from '@solana/web3.js';
import { TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID, createMint, createAssociatedTokenAccount, mintTo } from '@solana/spl-token';
import { expect, assert } from 'chai';
import { AiAgent } from '../target/types/ai_agent';
import { Vault } from '../target/types/vault';

interface TestAccounts {
  admin: Keypair;
  user: Keypair;
  vault: Keypair;
  agent: Keypair;
  mint: PublicKey;
  adminTokenAccount: PublicKey;
  userTokenAccount: PublicKey;
  vaultTokenAccount: PublicKey;
}

interface AgentState {
  admin: PublicKey;
  vault: PublicKey;
  model: {
    name: string;
    version: string;
    parameters: Buffer;
  };
  strategy: {
    type: number;
    parameters: Buffer;
    riskLevel: number;
  };
  status: {
    isActive: boolean;
    lastExecution: BN;
    totalTrades: BN;
    successRate: number;
  };
  performance: {
    totalPnl: BN;
    maxDrawdown: number;
    sharpeRatio: number;
    winRate: number;
  };
  constraints: {
    maxPositionSize: BN;
    maxDailyTrades: number;
    allowedMarkets: number[];
    blacklistedTokens: PublicKey[];
  };
  fees: {
    performanceFee: number;
    managementFee: number;
    feeRecipient: PublicKey;
  };
}

describe('AI Agent Smart Contract', () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const aiAgentProgram = anchor.workspace.AiAgent as Program<AiAgent>;
  const vaultProgram = anchor.workspace.Vault as Program<Vault>;
  let accounts: TestAccounts;

  // Test configuration constants
  const INITIAL_MINT_AMOUNT = new BN(1_000_000 * 10 ** 6); // 1M tokens with 6 decimals
  const INITIAL_VAULT_DEPOSIT = new BN(100_000 * 10 ** 6); // 100K tokens
  const AGENT_PERFORMANCE_FEE = 2000; // 20%
  const AGENT_MANAGEMENT_FEE = 100; // 1%
  const MAX_POSITION_SIZE = new BN(50_000 * 10 ** 6); // 50K tokens
  const MAX_DAILY_TRADES = 10;
  const RISK_LEVEL = 3; // Medium risk

  beforeEach(async () => {
    // Initialize test accounts
    accounts = {
      admin: Keypair.generate(),
      user: Keypair.generate(),
      vault: Keypair.generate(),
      agent: Keypair.generate(),
      mint: PublicKey.default,
      adminTokenAccount: PublicKey.default,
      userTokenAccount: PublicKey.default,
      vaultTokenAccount: PublicKey.default,
    };

    // Fund accounts with SOL
    await Promise.all([
      provider.connection.requestAirdrop(accounts.admin.publicKey, 10 * anchor.web3.LAMPORTS_PER_SOL),
      provider.connection.requestAirdrop(accounts.user.publicKey, 10 * anchor.web3.LAMPORTS_PER_SOL),
    ]);

    // Create test token mint
    accounts.mint = await createMint(
      provider.connection,
      accounts.admin,
      accounts.admin.publicKey,
      null,
      6
    );

    // Create associated token accounts
    accounts.adminTokenAccount = await createAssociatedTokenAccount(
      provider.connection,
      accounts.admin,
      accounts.mint,
      accounts.admin.publicKey
    );

    accounts.userTokenAccount = await createAssociatedTokenAccount(
      provider.connection,
      accounts.admin,
      accounts.mint,
      accounts.user.publicKey
    );

    // Mint tokens to admin
    await mintTo(
      provider.connection,
      accounts.admin,
      accounts.mint,
      accounts.adminTokenAccount,
      accounts.admin,
      INITIAL_MINT_AMOUNT.toNumber()
    );

    // Initialize vault
    const vaultPda = PublicKey.findProgramAddressSync(
      [Buffer.from('vault'), accounts.admin.publicKey.toBuffer()],
      vaultProgram.programId
    )[0];

    accounts.vaultTokenAccount = await createAssociatedTokenAccount(
      provider.connection,
      accounts.admin,
      accounts.mint,
      vaultPda,
      true
    );

    await vaultProgram.methods
      .initialize({
        name: 'Test Vault',
        description: 'Vault for AI agent testing',
        feeRecipient: accounts.admin.publicKey,
        performanceFee: 1000,
        managementFee: 50,
        minDeposit: new BN(1000 * 10 ** 6),
        maxCapacity: new BN(1_000_000 * 10 ** 6),
      })
      .accounts({
        vault: vaultPda,
        admin: accounts.admin.publicKey,
        mint: accounts.mint,
        vaultTokenAccount: accounts.vaultTokenAccount,
        tokenProgram: TOKEN_PROGRAM_ID,
        associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        systemProgram: SystemProgram.programId,
        rent: SYSVAR_RENT_PUBKEY,
      })
      .signers([accounts.admin])
      .rpc();
  });

  describe('Agent Initialization', () => {
    it('Should initialize AI agent with correct parameters', async () => {
      const [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      const modelParams = Buffer.from(JSON.stringify({
        architecture: 'transformer',
        layers: 12,
        hiddenSize: 768,
        attentionHeads: 12,
        dropout: 0.1,
      }));

      const strategyParams = Buffer.from(JSON.stringify({
        indicators: ['sma', 'ema', 'rsi', 'macd'],
        timeframes: ['1m', '5m', '15m', '1h'],
        entryThreshold: 0.7,
        exitThreshold: 0.3,
      }));

      await aiAgentProgram.methods
        .initialize({
          model: {
            name: 'Ariadne-v1.0',
            version: '1.0.0',
            parameters: modelParams,
          },
          strategy: {
            type: 1, // Momentum strategy
            parameters: strategyParams,
            riskLevel: RISK_LEVEL,
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1, 2, 3], // SPL tokens, derivatives, perpetuals
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      
      expect(agent.admin.toString()).to.equal(accounts.admin.publicKey.toString());
      expect(agent.vault.toString()).to.equal(accounts.vault.publicKey.toString());
      expect(agent.model.name).to.equal('Ariadne-v1.0');
      expect(agent.strategy.type).to.equal(1);
      expect(agent.strategy.riskLevel).to.equal(RISK_LEVEL);
      expect(agent.fees.performanceFee).to.equal(AGENT_PERFORMANCE_FEE);
      expect(agent.constraints.maxDailyTrades).to.equal(MAX_DAILY_TRADES);
      expect(agent.status.isActive).to.be.false;
    });

    it('Should fail with invalid parameters', async () => {
      const [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      try {
        await aiAgentProgram.methods
          .initialize({
            model: {
              name: '',
              version: '1.0.0',
              parameters: Buffer.alloc(0),
            },
            strategy: {
              type: 0,
              parameters: Buffer.alloc(0),
              riskLevel: 10, // Invalid risk level
            },
            constraints: {
              maxPositionSize: new BN(0),
              maxDailyTrades: 0,
              allowedMarkets: [],
              blacklistedTokens: [],
            },
            fees: {
              performanceFee: 10000, // 100% fee is invalid
              managementFee: 0,
              feeRecipient: accounts.admin.publicKey,
            },
          })
          .accounts({
            agent: agentPda,
            vault: accounts.vault.publicKey,
            admin: accounts.admin.publicKey,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();
        assert.fail('Should have failed with invalid parameters');
      } catch (error) {
        expect(error.message).to.include('InvalidParameters');
      }
    });
  });

  describe('Agent Configuration', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      // Initialize agent
      const modelParams = Buffer.from(JSON.stringify({ model: 'basic' }));
      const strategyParams = Buffer.from(JSON.stringify({ strategy: 'momentum' }));

      await aiAgentProgram.methods
        .initialize({
          model: { name: 'Test', version: '1.0.0', parameters: modelParams },
          strategy: { type: 1, parameters: strategyParams, riskLevel: 3 },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should update agent strategy', async () => {
      const newStrategyParams = Buffer.from(JSON.stringify({
        indicators: ['bollinger', 'stochastic'],
        timeframes: ['4h', '1d'],
        entryThreshold: 0.8,
        exitThreshold: 0.2,
      }));

      await aiAgentProgram.methods
        .updateStrategy({
          type: 2, // Mean reversion strategy
          parameters: newStrategyParams,
          riskLevel: 2,
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.strategy.type).to.equal(2);
      expect(agent.strategy.riskLevel).to.equal(2);
    });

    it('Should update agent constraints', async () => {
      const newMaxPosition = new BN(75_000 * 10 ** 6);
      const newMaxTrades = 15;

      await aiAgentProgram.methods
        .updateConstraints({
          maxPositionSize: newMaxPosition,
          maxDailyTrades: newMaxTrades,
          allowedMarkets: [1, 2],
          blacklistedTokens: [accounts.mint],
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.constraints.maxPositionSize.toString()).to.equal(newMaxPosition.toString());
      expect(agent.constraints.maxDailyTrades).to.equal(newMaxTrades);
      expect(agent.constraints.allowedMarkets).to.deep.equal([1, 2]);
    });

    it('Should update fee structure', async () => {
      const newPerformanceFee = 1500; // 15%
      const newManagementFee = 200; // 2%

      await aiAgentProgram.methods
        .updateFees({
          performanceFee: newPerformanceFee,
          managementFee: newManagementFee,
          feeRecipient: accounts.user.publicKey,
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.fees.performanceFee).to.equal(newPerformanceFee);
      expect(agent.fees.managementFee).to.equal(newManagementFee);
      expect(agent.fees.feeRecipient.toString()).to.equal(accounts.user.publicKey.toString());
    });

    it('Should prevent unauthorized updates', async () => {
      try {
        await aiAgentProgram.methods
          .updateStrategy({
            type: 3,
            parameters: Buffer.alloc(10),
            riskLevel: 4,
          })
          .accounts({
            agent: agentPda,
            admin: accounts.user.publicKey, // Wrong admin
          })
          .signers([accounts.user])
          .rpc();
        assert.fail('Should have failed with unauthorized access');
      } catch (error) {
        expect(error.message).to.include('ConstraintRaw');
      }
    });
  });

  describe('Agent Execution', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      // Initialize agent with basic parameters
      await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Ariadne-v1.0', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ layers: 12 })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ momentum: true })), 
            riskLevel: 3 
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1, 2, 3],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should activate agent', async () => {
      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.status.isActive).to.be.true;
    });

    it('Should deactivate agent', async () => {
      // First activate
      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      // Then deactivate
      await aiAgentProgram.methods
        .deactivate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.status.isActive).to.be.false;
    });

    it('Should execute trade signal', async () => {
      // Activate agent first
      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const signal = {
        action: 1, // Buy
        asset: accounts.mint,
        amount: new BN(1000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300, // 5 minutes
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .executeSignal(signal)
        .accounts({
          agent: agentPda,
          execution: executionPda,
          asset: accounts.mint,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.status.totalTrades.toNumber()).to.equal(1);
    });

    it('Should respect daily trade limits', async () => {
      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      // Execute maximum allowed trades
      for (let i = 0; i < MAX_DAILY_TRADES; i++) {
        const signal = {
          action: 1,
          asset: accounts.mint,
          amount: new BN(1000 * 10 ** 6),
          targetPrice: new BN(100 * 10 ** 6),
          stopLoss: new BN(90 * 10 ** 6),
          takeProfit: new BN(110 * 10 ** 6),
          confidence: 85,
          timeframe: 300,
        };

        const [executionPda] = PublicKey.findProgramAddressSync(
          [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer(), Buffer.from([i])],
          aiAgentProgram.programId
        );

        await aiAgentProgram.methods
          .executeSignal(signal)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();
      }

      // Try to execute one more trade (should fail)
      try {
        const signal = {
          action: 1,
          asset: accounts.mint,
          amount: new BN(1000 * 10 ** 6),
          targetPrice: new BN(100 * 10 ** 6),
          stopLoss: new BN(90 * 10 ** 6),
          takeProfit: new BN(110 * 10 ** 6),
          confidence: 85,
          timeframe: 300,
        };

        const [executionPda] = PublicKey.findProgramAddressSync(
          [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer(), Buffer.from([MAX_DAILY_TRADES])],
          aiAgentProgram.programId
        );

        await aiAgentProgram.methods
          .executeSignal(signal)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();
        assert.fail('Should have failed due to daily trade limit');
      } catch (error) {
        expect(error.message).to.include('DailyTradeLimitExceeded');
      }
    });
  });

  describe('Performance Tracking', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Ariadne-v1.0', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ layers: 12 })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ momentum: true })), 
            riskLevel: 3 
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1, 2, 3],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should track trade outcomes', async () => {
      // Execute a profitable trade
      const profitableTrade = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(10000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 90,
        timeframe: 300,
      };

      const [executionPda1] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer(), Buffer.from([0])],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .executeSignal(profitableTrade)
        .accounts({
          agent: agentPda,
          execution: executionPda1,
          asset: accounts.mint,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      // Update trade outcome as profitable
      await aiAgentProgram.methods
        .updateTradeOutcome({
          executionId: 0,
          actualPrice: new BN(110 * 10 ** 6), // Hit take profit
          pnl: new BN(1000 * 10 ** 6), // 10% profit
          isSuccess: true,
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.performance.totalPnl.toNumber()).to.equal(1000 * 10 ** 6);
      expect(agent.performance.winRate).to.be.above(0);
    });

    it('Should calculate Sharpe ratio', async () => {
      // Execute multiple trades with varying outcomes
      const trades = [
        { pnl: 1000 * 10 ** 6, success: true },
        { pnl: -500 * 10 ** 6, success: false },
        { pnl: 2000 * 10 ** 6, success: true },
        { pnl: -200 * 10 ** 6, success: false },
        { pnl: 1500 * 10 ** 6, success: true },
      ];

      for (let i = 0; i < trades.length; i++) {
        const signal = {
          action: 1,
          asset: accounts.mint,
          amount: new BN(10000 * 10 ** 6),
          targetPrice: new BN(100 * 10 ** 6),
          stopLoss: new BN(90 * 10 ** 6),
          takeProfit: new BN(110 * 10 ** 6),
          confidence: 80,
          timeframe: 300,
        };

        const [executionPda] = PublicKey.findProgramAddressSync(
          [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer(), Buffer.from([i])],
          aiAgentProgram.programId
        );

        await aiAgentProgram.methods
          .executeSignal(signal)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();

        await aiAgentProgram.methods
          .updateTradeOutcome({
            executionId: i,
            actualPrice: new BN(105 * 10 ** 6),
            pnl: new BN(trades[i].pnl),
            isSuccess: trades[i].success,
          })
          .accounts({
            agent: agentPda,
            admin: accounts.admin.publicKey,
          })
          .signers([accounts.admin])
          .rpc();
      }

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      // Maximum drawdown should be around -40% (4000 loss on starting balance)
      expect(agent.performance.maxDrawdown).to.be.below(-0.3);
    });
  });

  describe('Risk Management', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Ariadne-v1.0', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ layers: 12 })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ momentum: true })), 
            riskLevel: 2 // Conservative risk level
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should enforce position size limits', async () => {
      const oversizedTrade = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(100_000 * 10 ** 6), // Exceeds max position size
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      try {
        await aiAgentProgram.methods
          .executeSignal(oversizedTrade)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();
        assert.fail('Should have failed due to position size limit');
      } catch (error) {
        expect(error.message).to.include('PositionSizeExceeded');
      }
    });

    it('Should respect blacklisted tokens', async () => {
      // Add token to blacklist
      await aiAgentProgram.methods
        .updateConstraints({
          maxPositionSize: MAX_POSITION_SIZE,
          maxDailyTrades: MAX_DAILY_TRADES,
          allowedMarkets: [1],
          blacklistedTokens: [accounts.mint],
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const signal = {
        action: 1,
        asset: accounts.mint, // Blacklisted asset
        amount: new BN(1000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      try {
        await aiAgentProgram.methods
          .executeSignal(signal)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();
        assert.fail('Should have failed due to blacklisted token');
      } catch (error) {
        expect(error.message).to.include('BlacklistedAsset');
      }
    });

    it('Should enforce market type restrictions', async () => {
      // Update to only allow spot markets (type 1)
      await aiAgentProgram.methods
        .updateConstraints({
          maxPositionSize: MAX_POSITION_SIZE,
          maxDailyTrades: MAX_DAILY_TRADES,
          allowedMarkets: [1], // Only spot markets
          blacklistedTokens: [],
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      // Try to execute a derivatives trade (market type 2)
      const signal = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(1000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300,
        marketType: 2, // Derivatives
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      try {
        await aiAgentProgram.methods
          .executeSignal(signal)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();
        assert.fail('Should have failed due to market type restriction');
      } catch (error) {
        expect(error.message).to.include('InvalidMarketType');
      }
    });

    it('Should implement risk-based position sizing', async () => {
      // High risk level should reduce position size
      await aiAgentProgram.methods
        .updateStrategy({
          type: 1,
          parameters: Buffer.from(JSON.stringify({ momentum: true })),
          riskLevel: 5, // High risk
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const signal = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(50_000 * 10 ** 6), // Max position size
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 50, // Low confidence
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .executeSignal(signal)
        .accounts({
          agent: agentPda,
          execution: executionPda,
          asset: accounts.mint,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      // Check that position size was adjusted based on risk
      const execution = await aiAgentProgram.account.execution.fetch(executionPda);
      expect(execution.actualAmount.lt(signal.amount)).to.be.true;
    });
  });

  describe('Fee Management', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Ariadne-v1.0', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ layers: 12 })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ momentum: true })), 
            riskLevel: 3 
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1, 2, 3],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should calculate performance fees correctly', async () => {
      // Execute profitable trade
      const signal = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(10_000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(120 * 10 ** 6),
        confidence: 90,
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .executeSignal(signal)
        .accounts({
          agent: agentPda,
          execution: executionPda,
          asset: accounts.mint,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      // Update with 20% profit
      const profit = new BN(2_000 * 10 ** 6);
      await aiAgentProgram.methods
        .updateTradeOutcome({
          executionId: 0,
          actualPrice: new BN(120 * 10 ** 6),
          pnl: profit,
          isSuccess: true,
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      // Calculate performance fee
      const expectedFee = profit.muln(AGENT_PERFORMANCE_FEE).divn(10000);
      
      await aiAgentProgram.methods
        .collectFees()
        .accounts({
          agent: agentPda,
          feeRecipient: accounts.admin.publicKey,
          feeAccount: accounts.adminTokenAccount,
          tokenProgram: TOKEN_PROGRAM_ID,
        })
        .signers([accounts.admin])
        .rpc();

      // Verify fee collection
      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.performance.totalPnl.sub(agent.performance.accumulatedFees).toString())
        .to.equal(profit.sub(expectedFee).toString());
    });

    it('Should handle management fees over time', async () => {
      // Simulate time passage for management fee calculation
      const timeElapsed = 86400 * 30; // 30 days in seconds
      
      await aiAgentProgram.methods
        .updateManagementFee(timeElapsed)
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.fees.accumulatedManagementFees.gt(new BN(0))).to.be.true;
    });

    it('Should only allow fee collection by fee recipient', async () => {
      // Try to collect fees with wrong recipient
      try {
        await aiAgentProgram.methods
          .collectFees()
          .accounts({
            agent: agentPda,
            feeRecipient: accounts.user.publicKey, // Wrong recipient
            feeAccount: accounts.userTokenAccount,
            tokenProgram: TOKEN_PROGRAM_ID,
          })
          .signers([accounts.user])
          .rpc();
        assert.fail('Should have failed with unauthorized fee collection');
      } catch (error) {
        expect(error.message).to.include('UnauthorizedFeeCollection');
      }
    });
  });

  describe('Emergency Controls', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Ariadne-v1.0', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ layers: 12 })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ momentum: true })), 
            riskLevel: 3 
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1, 2, 3],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should pause agent in emergency', async () => {
      await aiAgentProgram.methods
        .emergencyPause()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.status.isEmergencyPaused).to.be.true;

      // Try to execute trade while paused (should fail)
      const signal = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(1000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      try {
        await aiAgentProgram.methods
          .executeSignal(signal)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();
        assert.fail('Should have failed due to emergency pause');
      } catch (error) {
        expect(error.message).to.include('EmergencyPaused');
      }
    });

    it('Should unpause after emergency', async () => {
      // Pause
      await aiAgentProgram.methods
        .emergencyPause()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      // Unpause
      await aiAgentProgram.methods
        .emergencyUnpause()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.status.isEmergencyPaused).to.be.false;

      // Should be able to execute trades again
      const signal = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(1000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .executeSignal(signal)
        .accounts({
          agent: agentPda,
          execution: executionPda,
          asset: accounts.mint,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      const updatedAgent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(updatedAgent.status.totalTrades.toNumber()).to.equal(1);
    });

    it('Should manually close positions in emergency', async () => {
      // Execute a trade first
      const signal = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(10_000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .executeSignal(signal)
        .accounts({
          agent: agentPda,
          execution: executionPda,
          asset: accounts.mint,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      // Emergency close position
      await aiAgentProgram.methods
        .emergencyClosePosition(0) // Close first position
        .accounts({
          agent: agentPda,
          execution: executionPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const execution = await aiAgentProgram.account.execution.fetch(executionPda);
      expect(execution.status.isClosed).to.be.true;
      expect(execution.status.emergencyClosed).to.be.true;
    });
  });

  describe('Model Updates', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Ariadne-v1.0', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ layers: 12 })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ momentum: true })), 
            riskLevel: 3 
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1, 2, 3],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should update model parameters', async () => {
      const newModelParams = Buffer.from(JSON.stringify({
        architecture: 'transformer',
        layers: 24,
        hiddenSize: 1024,
        attentionHeads: 16,
        dropout: 0.05,
      }));

      await aiAgentProgram.methods
        .updateModel({
          name: 'Ariadne-v2.0',
          version: '2.0.0',
          parameters: newModelParams,
        })
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.model.name).to.equal('Ariadne-v2.0');
      expect(agent.model.version).to.equal('2.0.0');
    });

    it('Should validate model parameters', async () => {
      const invalidParams = Buffer.alloc(100_000); // Too large

      try {
        await aiAgentProgram.methods
          .updateModel({
            name: 'Invalid-Model',
            version: '0.0.1',
            parameters: invalidParams,
          })
          .accounts({
            agent: agentPda,
            admin: accounts.admin.publicKey,
          })
          .signers([accounts.admin])
          .rpc();
        assert.fail('Should have failed with invalid model parameters');
      } catch (error) {
        expect(error.message).to.include('InvalidModelParameters');
      }
    });
  });

  describe('Event Emission', () => {
    let agentPda: PublicKey;

    beforeEach(async () => {
      [agentPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('ai-agent'), accounts.admin.publicKey.toBuffer(), accounts.vault.publicKey.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Ariadne-v1.0', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ layers: 12 })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ momentum: true })), 
            riskLevel: 3 
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1, 2, 3],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: agentPda,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();
    });

    it('Should emit events on agent initialization', async () => {
      const txSignature = await aiAgentProgram.methods
        .initialize({
          model: { 
            name: 'Test-Agent', 
            version: '1.0.0', 
            parameters: Buffer.from(JSON.stringify({ test: true })) 
          },
          strategy: { 
            type: 1, 
            parameters: Buffer.from(JSON.stringify({ strategy: 'test' })), 
            riskLevel: 3 
          },
          constraints: {
            maxPositionSize: MAX_POSITION_SIZE,
            maxDailyTrades: MAX_DAILY_TRADES,
            allowedMarkets: [1],
            blacklistedTokens: [],
          },
          fees: {
            performanceFee: AGENT_PERFORMANCE_FEE,
            managementFee: AGENT_MANAGEMENT_FEE,
            feeRecipient: accounts.admin.publicKey,
          },
        })
        .accounts({
          agent: Keypair.generate().publicKey,
          vault: accounts.vault.publicKey,
          admin: accounts.admin.publicKey,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      // Fetch and verify events
      const events = await aiAgentProgram.account.eventLog.all();
      expect(events.length).to.be.above(0);
      expect(events[events.length - 1].account.eventType).to.equal('AgentInitialized');
    });

    it('Should emit events on trade execution', async () => {
      await aiAgentProgram.methods
        .activate()
        .accounts({
          agent: agentPda,
          admin: accounts.admin.publicKey,
        })
        .signers([accounts.admin])
        .rpc();

      const signal = {
        action: 1,
        asset: accounts.mint,
        amount: new BN(1000 * 10 ** 6),
        targetPrice: new BN(100 * 10 ** 6),
        stopLoss: new BN(90 * 10 ** 6),
        takeProfit: new BN(110 * 10 ** 6),
        confidence: 85,
        timeframe: 300,
      };

      const [executionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer()],
        aiAgentProgram.programId
      );

      await aiAgentProgram.methods
        .executeSignal(signal)
        .accounts({
          agent: agentPda,
          execution: executionPda,
          asset: accounts.mint,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .signers([accounts.admin])
        .rpc();

      const events = await aiAgentProgram.account.eventLog.all();
      const tradeEvents = events.filter(e => e.account.eventType === 'TradeExecuted');
      expect(tradeEvents.length).to.be.above(0);
    });
  });
}); BN(trades[i].pnl),
            isSuccess: trades[i].success,
          })
          .accounts({
            agent: agentPda,
            admin: accounts.admin.publicKey,
          })
          .signers([accounts.admin])
          .rpc();
      }

      const agent = await aiAgentProgram.account.agent.fetch(agentPda);
      expect(agent.performance.sharpeRatio).to.be.above(0);
      expect(agent.performance.winRate).to.equal(0.6); // 3 out of 5 trades successful
    });

    it('Should track maximum drawdown', async () => {
      const trades = [
        { pnl: 1000 * 10 ** 6, success: true },
        { pnl: -3000 * 10 ** 6, success: false }, // Big loss causing drawdown
        { pnl: -1000 * 10 ** 6, success: false }, // Continue drawdown
        { pnl: 2000 * 10 ** 6, success: true }, // Recovery
      ];

      for (let i = 0; i < trades.length; i++) {
        const signal = {
          action: 1,
          asset: accounts.mint,
          amount: new BN(10000 * 10 ** 6),
          targetPrice: new BN(100 * 10 ** 6),
          stopLoss: new BN(90 * 10 ** 6),
          takeProfit: new BN(110 * 10 ** 6),
          confidence: 80,
          timeframe: 300,
        };

        const [executionPda] = PublicKey.findProgramAddressSync(
          [Buffer.from('execution'), agentPda.toBuffer(), accounts.mint.toBuffer(), Buffer.from([i])],
          aiAgentProgram.programId
        );

        await aiAgentProgram.methods
          .executeSignal(signal)
          .accounts({
            agent: agentPda,
            execution: executionPda,
            asset: accounts.mint,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .signers([accounts.admin])
          .rpc();

        await aiAgentProgram.methods
          .updateTradeOutcome({
            executionId: i,
            actualPrice: new BN(105 * 10 ** 6),
            pnl: new await aiAgentProgram.methods
  .updateTradeOutcome({
    executionId: i,
    actualPrice: new BN(105 * 10 ** 6),
    pnl: new BN(15 * 10 ** 6),
  })
  .accounts({
    agent: agentPda,
    execution: executionPda,
    asset: accounts.mint,
    systemProgram: SystemProgram.programId,
    rent: SYSVAR_RENT_PUBKEY,
  })
  .signers([accounts.admin])
  .rpc();
  interface AgentState {
  admin: PublicKey;
  vault: PublicKey;
  model: {
    name: string;
    version: string;
  };
}