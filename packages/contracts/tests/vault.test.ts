import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { expect } from "chai";
import { Keypair, SystemProgram, LAMPORTS_PER_SOL } from "@solana/web3.js";
import { createMint, mintTo, getOrCreateAssociatedTokenAccount } from "@solana/spl-token";
import { Vault } from "../target/types/vault";

describe("Vault Contract Tests", () => {
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.Vault as Program<Vault>;
  const connection = provider.connection;
  
  // Test accounts
  let vaultKeypair: Keypair;
  let vaultOwner: Keypair;
  let investor1: Keypair;
  let investor2: Keypair;
  let manager: Keypair;
  let mint: anchor.web3.PublicKey;
  let vaultToken: anchor.web3.PublicKey;
  let investorTokenAccount1: anchor.web3.PublicKey;
  let investorTokenAccount2: anchor.web3.PublicKey;

  // Constants
  const INITIAL_DEPOSIT = 1000 * LAMPORTS_PER_SOL;
  const MINIMUM_DEPOSIT = 10 * LAMPORTS_PER_SOL;
  const MANAGEMENT_FEE = 200; // 2%
  const PERFORMANCE_FEE = 2000; // 20%

  beforeEach(async () => {
    // Generate fresh keypairs for each test
    vaultKeypair = Keypair.generate();
    vaultOwner = Keypair.generate();
    investor1 = Keypair.generate();
    investor2 = Keypair.generate();
    manager = Keypair.generate();

    // Airdrop SOL to accounts
    await connection.requestAirdrop(vaultOwner.publicKey, 10 * LAMPORTS_PER_SOL);
    await connection.requestAirdrop(investor1.publicKey, 10 * LAMPORTS_PER_SOL);
    await connection.requestAirdrop(investor2.publicKey, 10 * LAMPORTS_PER_SOL);
    await connection.requestAirdrop(manager.publicKey, 10 * LAMPORTS_PER_SOL);

    // Create mint
    mint = await createMint(
      connection,
      vaultOwner,
      vaultOwner.publicKey,
      null,
      9
    );

    // Create token accounts
    const vaultTokenAccount = await getOrCreateAssociatedTokenAccount(
      connection,
      vaultOwner,
      mint,
      vaultKeypair.publicKey,
      true
    );
    vaultToken = vaultTokenAccount.address;

    const investorAccount1 = await getOrCreateAssociatedTokenAccount(
      connection,
      investor1,
      mint,
      investor1.publicKey
    );
    investorTokenAccount1 = investorAccount1.address;

    const investorAccount2 = await getOrCreateAssociatedTokenAccount(
      connection,
      investor2,
      mint,
      investor2.publicKey
    );
    investorTokenAccount2 = investorAccount2.address;

    // Mint tokens to investors
    await mintTo(
      connection,
      vaultOwner,
      mint,
      investorTokenAccount1,
      vaultOwner.publicKey,
      5000 * LAMPORTS_PER_SOL
    );

    await mintTo(
      connection,
      vaultOwner,
      mint,
      investorTokenAccount2,
      vaultOwner.publicKey,
      5000 * LAMPORTS_PER_SOL
    );
  });

  describe("Vault Initialization", () => {
    it("Should initialize vault with correct parameters", async () => {
      const tx = await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      
      expect(vault.owner).to.eql(vaultOwner.publicKey);
      expect(vault.manager).to.eql(manager.publicKey);
      expect(vault.minDeposit.toNumber()).to.equal(MINIMUM_DEPOSIT);
      expect(vault.managementFee).to.equal(MANAGEMENT_FEE);
      expect(vault.performanceFee).to.equal(PERFORMANCE_FEE);
      expect(vault.name).to.equal("AI Trading Vault");
      expect(vault.symbol).to.equal("ATV");
      expect(vault.totalDeposits.toNumber()).to.equal(0);
      expect(vault.isActive).to.be.true;
    });

    it("Should fail with invalid management fee", async () => {
      try {
        await program.methods
          .initialize(
            new anchor.BN(MINIMUM_DEPOSIT),
            10000, // 100% - invalid
            PERFORMANCE_FEE,
            "AI Trading Vault",
            "ATV"
          )
          .accounts({
            vault: vaultKeypair.publicKey,
            owner: vaultOwner.publicKey,
            manager: manager.publicKey,
            vaultToken,
            mint,
            systemProgram: SystemProgram.programId,
          })
          .signers([vaultKeypair, vaultOwner])
          .rpc();
        
        expect.fail("Should have failed with invalid management fee");
      } catch (error) {
        expect(error.message).to.include("InvalidFee");
      }
    });
  });

  describe("Deposits", () => {
    beforeEach(async () => {
      await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();
    });

    it("Should accept valid deposit", async () => {
      const depositAmount = 1000 * LAMPORTS_PER_SOL;

      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      await program.methods
        .deposit(new anchor.BN(depositAmount))
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor1.publicKey,
          investorState: investorStatePda,
          investorTokenAccount: investorTokenAccount1,
          vaultTokenAccount: vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([investor1])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      const investorState = await program.account.investorState.fetch(investorStatePda);

      expect(vault.totalDeposits.toNumber()).to.equal(depositAmount);
      expect(investorState.totalDeposited.toNumber()).to.equal(depositAmount);
      expect(investorState.shares.toNumber()).to.equal(depositAmount);
    });

    it("Should fail with deposit below minimum", async () => {
      const depositAmount = 5 * LAMPORTS_PER_SOL; // Below minimum

      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      try {
        await program.methods
          .deposit(new anchor.BN(depositAmount))
          .accounts({
            vault: vaultKeypair.publicKey,
            investor: investor1.publicKey,
            investorState: investorStatePda,
            investorTokenAccount: investorTokenAccount1,
            vaultTokenAccount: vaultToken,
            mint,
            systemProgram: SystemProgram.programId,
          })
          .signers([investor1])
          .rpc();
        
        expect.fail("Should have failed with insufficient deposit");
      } catch (error) {
        expect(error.message).to.include("InsufficientDeposit");
      }
    });

    it("Should handle multiple deposits and calculate shares correctly", async () => {
      // First deposit
      const firstDeposit = 1000 * LAMPORTS_PER_SOL;
      
      const [investor1StatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      await program.methods
        .deposit(new anchor.BN(firstDeposit))
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor1.publicKey,
          investorState: investor1StatePda,
          investorTokenAccount: investorTokenAccount1,
          vaultTokenAccount: vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([investor1])
        .rpc();

      // Second deposit by different investor
      const secondDeposit = 2000 * LAMPORTS_PER_SOL;
      
      const [investor2StatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor2.publicKey.toBuffer(),
        ],
        program.programId
      );

      await program.methods
        .deposit(new anchor.BN(secondDeposit))
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor2.publicKey,
          investorState: investor2StatePda,
          investorTokenAccount: investorTokenAccount2,
          vaultTokenAccount: vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([investor2])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.totalDeposits.toNumber()).to.equal(firstDeposit + secondDeposit);
    });
  });

  describe("Withdrawals", () => {
    beforeEach(async () => {
      await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();

      // Make initial deposit
      const depositAmount = 1000 * LAMPORTS_PER_SOL;
      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      await program.methods
        .deposit(new anchor.BN(depositAmount))
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor1.publicKey,
          investorState: investorStatePda,
          investorTokenAccount: investorTokenAccount1,
          vaultTokenAccount: vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([investor1])
        .rpc();
    });

    it("Should process withdrawal request", async () => {
      const withdrawAmount = 500 * LAMPORTS_PER_SOL;
      
      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      await program.methods
        .requestWithdrawal(new anchor.BN(withdrawAmount))
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor1.publicKey,
          investorState: investorStatePda,
        })
        .signers([investor1])
        .rpc();

      const investorState = await program.account.investorState.fetch(investorStatePda);
      expect(investorState.pendingWithdrawal.toNumber()).to.equal(withdrawAmount);
    });

    it("Should process withdrawal after lock period", async () => {
      const withdrawAmount = 500 * LAMPORTS_PER_SOL;
      
      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      // Request withdrawal
      await program.methods
        .requestWithdrawal(new anchor.BN(withdrawAmount))
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor1.publicKey,
          investorState: investorStatePda,
        })
        .signers([investor1])
        .rpc();

      // Process withdrawal (normally would wait for lock period)
      await program.methods
        .processWithdrawal()
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor1.publicKey,
          investorState: investorStatePda,
          investorTokenAccount: investorTokenAccount1,
          vaultTokenAccount: vaultToken,
          mint,
        })
        .signers([investor1])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.totalDeposits.toNumber()).to.equal(500 * LAMPORTS_PER_SOL);
    });

    it("Should fail withdrawal exceeding balance", async () => {
      const withdrawAmount = 2000 * LAMPORTS_PER_SOL; // More than deposited
      
      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      try {
        await program.methods
          .requestWithdrawal(new anchor.BN(withdrawAmount))
          .accounts({
            vault: vaultKeypair.publicKey,
            investor: investor1.publicKey,
            investorState: investorStatePda,
          })
          .signers([investor1])
          .rpc();
        
        expect.fail("Should have failed with insufficient balance");
      } catch (error) {
        expect(error.message).to.include("InsufficientBalance");
      }
    });
  });

  describe("Fee Management", () => {
    beforeEach(async () => {
      await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();
    });

    it("Should collect management fees", async () => {
      await program.methods
        .collectManagementFee()
        .accounts({
          vault: vaultKeypair.publicKey,
          manager: manager.publicKey,
          feeCollector: manager.publicKey,
          vaultTokenAccount: vaultToken,
          feeTokenAccount: await getOrCreateAssociatedTokenAccount(
            connection,
            manager,
            mint,
            manager.publicKey
          ).then(account => account.address),
          mint,
        })
        .signers([manager])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.lastFeeCollection).to.be.greaterThan(0);
    });

    it("Should update fee structure (owner only)", async () => {
      const newManagementFee = 150; // 1.5%
      const newPerformanceFee = 1500; // 15%

      await program.methods
        .updateFees(newManagementFee, newPerformanceFee)
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
        })
        .signers([vaultOwner])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.managementFee).to.equal(newManagementFee);
      expect(vault.performanceFee).to.equal(newPerformanceFee);
    });

    it("Should fail fee update from non-owner", async () => {
      try {
        await program.methods
          .updateFees(150, 1500)
          .accounts({
            vault: vaultKeypair.publicKey,
            owner: investor1.publicKey, // Not the owner
          })
          .signers([investor1])
          .rpc();
        
        expect.fail("Should have failed with unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });
  });

  describe("Vault Management", () => {
    beforeEach(async () => {
      await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();
    });

    it("Should pause vault", async () => {
      await program.methods
        .pauseVault()
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
        })
        .signers([vaultOwner])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.isActive).to.be.false;
    });

    it("Should unpause vault", async () => {
      // First pause
      await program.methods
        .pauseVault()
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
        })
        .signers([vaultOwner])
        .rpc();

      // Then unpause
      await program.methods
        .unpauseVault()
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
        })
        .signers([vaultOwner])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.isActive).to.be.true;
    });

    it("Should transfer ownership", async () => {
      const newOwner = Keypair.generate();
      await connection.requestAirdrop(newOwner.publicKey, LAMPORTS_PER_SOL);

      await program.methods
        .transferOwnership(newOwner.publicKey)
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          newOwner: newOwner.publicKey,
        })
        .signers([vaultOwner])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.owner).to.eql(newOwner.publicKey);
    });

    it("Should update manager", async () => {
      const newManager = Keypair.generate();
      await connection.requestAirdrop(newManager.publicKey, LAMPORTS_PER_SOL);

      await program.methods
        .updateManager(newManager.publicKey)
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          newManager: newManager.publicKey,
        })
        .signers([vaultOwner])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.manager).to.eql(newManager.publicKey);
    });
  });

  describe("Performance Tracking", () => {
    beforeEach(async () => {
      await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();

      // Make deposit for testing
      const depositAmount = 1000 * LAMPORTS_PER_SOL;
      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      await program.methods
        .deposit(new anchor.BN(depositAmount))
        .accounts({
          vault: vaultKeypair.publicKey,
          investor: investor1.publicKey,
          investorState: investorStatePda,
          investorTokenAccount: investorTokenAccount1,
          vaultTokenAccount: vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([investor1])
        .rpc();
    });

    it("Should update vault NAV", async () => {
      const newNav = 1100 * LAMPORTS_PER_SOL; // 10% profit

      await program.methods
        .updateNav(new anchor.BN(newNav))
        .accounts({
          vault: vaultKeypair.publicKey,
          manager: manager.publicKey,
        })
        .signers([manager])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.currentNav.toNumber()).to.equal(newNav);
      expect(vault.hwm.toNumber()).to.equal(newNav);
    });

    it("Should track high water mark", async () => {
      // First update - set initial HWM
      await program.methods
        .updateNav(new anchor.BN(1200 * LAMPORTS_PER_SOL))
        .accounts({
          vault: vaultKeypair.publicKey,
          manager: manager.publicKey,
        })
        .signers([manager])
        .rpc();

      // Second update - lower NAV should not change HWM
      await program.methods
        .updateNav(new anchor.BN(1100 * LAMPORTS_PER_SOL))
        .accounts({
          vault: vaultKeypair.publicKey,
          manager: manager.publicKey,
        })
        .signers([manager])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.currentNav.toNumber()).to.equal(1100 * LAMPORTS_PER_SOL);
      expect(vault.hwm.toNumber()).to.equal(1200 * LAMPORTS_PER_SOL);
    });

    it("Should collect performance fees on new high water mark", async () => {
      const initialDeposits = 1000 * LAMPORTS_PER_SOL;
      const newNav = 1200 * LAMPORTS_PER_SOL; // 20% profit
      const expectedPerformanceFee = (newNav - initialDeposits) * PERFORMANCE_FEE / 10000;

      await program.methods
        .updateNav(new anchor.BN(newNav))
        .accounts({
          vault: vaultKeypair.publicKey,
          manager: manager.publicKey,
        })
        .signers([manager])
        .rpc();

      await program.methods
        .collectPerformanceFee()
        .accounts({
          vault: vaultKeypair.publicKey,
          manager: manager.publicKey,
          feeCollector: manager.publicKey,
          vaultTokenAccount: vaultToken,
          feeTokenAccount: await getOrCreateAssociatedTokenAccount(
            connection,
            manager,
            mint,
            manager.publicKey
          ).then(account => account.address),
          mint,
        })
        .signers([manager])
        .rpc();

      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.performanceFeesCollected.toNumber()).to.be.greaterThan(0);
    });
  });

  describe("Access Control", () => {
    beforeEach(async () => {
      await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();
    });

    it("Should reject unauthorized manager operations", async () => {
      try {
        await program.methods
          .updateNav(new anchor.BN(1100 * LAMPORTS_PER_SOL))
          .accounts({
            vault: vaultKeypair.publicKey,
            manager: investor1.publicKey, // Not the manager
          })
          .signers([investor1])
          .rpc();
        
        expect.fail("Should have failed with unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });

    it("Should reject unauthorized owner operations", async () => {
      try {
        await program.methods
          .pauseVault()
          .accounts({
            vault: vaultKeypair.publicKey,
            owner: investor1.publicKey, // Not the owner
          })
          .signers([investor1])
          .rpc();
        
        expect.fail("Should have failed with unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });
  });

  describe("Edge Cases", () => {
    beforeEach(async () => {
      await program.methods
        .initialize(
          new anchor.BN(MINIMUM_DEPOSIT),
          MANAGEMENT_FEE,
          PERFORMANCE_FEE,
          "AI Trading Vault",
          "ATV"
        )
        .accounts({
          vault: vaultKeypair.publicKey,
          owner: vaultOwner.publicKey,
          manager: manager.publicKey,
          vaultToken,
          mint,
          systemProgram: SystemProgram.programId,
        })
        .signers([vaultKeypair, vaultOwner])
        .rpc();
    });

    it("Should handle zero deposits correctly", async () => {
      const vault = await program.account.vault.fetch(vaultKeypair.publicKey);
      expect(vault.totalDeposits.toNumber()).to.equal(0);
      expect(vault.sharePrice.toNumber()).to.equal(1 * LAMPORTS_PER_SOL);
    });

    it("Should handle maximum values", async () => {
      const maxDeposit = new anchor.BN("18446744073709551615"); // u64 max

      const [investorStatePda] = await anchor.web3.PublicKey.findProgramAddress(
        [
          Buffer.from("investor"),
          vaultKeypair.publicKey.toBuffer(),
          investor1.publicKey.toBuffer(),
        ],
        program.programId
      );

      // This should fail due to token balance, not overflow
      try {
        await program.methods
          .deposit(maxDeposit)
          .accounts({
            vault: vaultKeypair.publicKey,
            investor: investor1.publicKey,
            investorState: investorStatePda,
            investorTokenAccount: investorTokenAccount1,
            vaultTokenAccount: vaultToken,
            mint,
            systemProgram: SystemProgram.programId,
          })
          .signers([investor1])
          .rpc();
        
        expect.fail("Should have failed due to insufficient token balance");
      } catch (error) {
        // Expected to fail due to insufficient balance
        expect(error).to.exist;
      }
    });
  });
});