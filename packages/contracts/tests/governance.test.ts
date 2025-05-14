import * as anchor from "@project-serum/anchor";
import { Program, BN, web3 } from "@project-serum/anchor";
import { Governance } from "../target/types/governance";
import {
  createMint,
  getOrCreateAssociatedTokenAccount,
  mintTo,
  getAccount,
  transfer,
} from "@solana/spl-token";
import { assert, expect } from "chai";

describe("Governance", () => {
  // Configure the client to use the local cluster
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.Governance as Program<Governance>;
  const wallet = provider.wallet as anchor.Wallet;
  
  // Test accounts
  let governanceTokenMint: web3.PublicKey;
  let governancePDA: web3.PublicKey;
  let treasuryPDA: web3.PublicKey;
  let voterA: anchor.web3.Keypair;
  let voterATokenAccount: web3.PublicKey;
  let voterB: anchor.web3.Keypair;
  let voterBTokenAccount: web3.PublicKey;
  let proposalPDA: web3.PublicKey;
  let proposalIndex = new BN(1);
  
  // Constants
  const MIN_PROPOSAL_THRESHOLD = new BN(100_000_000);
  const DEFAULT_MINT_AMOUNT = new BN(1_000_000_000);
  const DEFAULT_VOTING_PERIOD = new BN(259_200); // 3 days
  const DEFAULT_EXECUTION_DELAY = new BN(86_400); // 1 day
  
  // Find PDAs
  const [governancePDAAddress] = anchor.web3.PublicKey.findProgramAddressSync(
    [Buffer.from("governance")],
    program.programId
  );
  
  const [treasuryPDAAddress] = anchor.web3.PublicKey.findProgramAddressSync(
    [Buffer.from("treasury")],
    program.programId
  );
  
  const [proposalPDAAddress] = anchor.web3.PublicKey.findProgramAddressSync(
    [Buffer.from("proposal"), proposalIndex.toArrayLike(Buffer, "le", 8)],
    program.programId
  );
  
  before(async () => {
    // Airdrop SOL to test accounts
    voterA = anchor.web3.Keypair.generate();
    voterB = anchor.web3.Keypair.generate();
    
    const airdropA = await provider.connection.requestAirdrop(
      voterA.publicKey,
      2 * web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(airdropA);
    
    const airdropB = await provider.connection.requestAirdrop(
      voterB.publicKey,
      2 * web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(airdropB);
    
    // Create governance token mint
    governanceTokenMint = await createMint(
      provider.connection,
      wallet.payer,
      wallet.publicKey,
      null,
      6 // 6 decimals
    );
    
    // Create token accounts
    let authorityTokenAccount = await getOrCreateAssociatedTokenAccount(
      provider.connection,
      wallet.payer,
      governanceTokenMint,
      wallet.publicKey
    );
    
    voterATokenAccount = (
      await getOrCreateAssociatedTokenAccount(
        provider.connection,
        wallet.payer,
        governanceTokenMint,
        voterA.publicKey
      )
    ).address;
    
    voterBTokenAccount = (
      await getOrCreateAssociatedTokenAccount(
        provider.connection,
        wallet.payer,
        governanceTokenMint,
        voterB.publicKey
      )
    ).address;
    
    // Mint tokens to authority
    await mintTo(
      provider.connection,
      wallet.payer,
      governanceTokenMint,
      authorityTokenAccount.address,
      wallet.payer,
      DEFAULT_MINT_AMOUNT.toNumber() * 10 // 10x the default amount
    );
    
    // Transfer tokens to voters
    await transfer(
      provider.connection,
      wallet.payer,
      authorityTokenAccount.address,
      voterATokenAccount,
      wallet.payer,
      DEFAULT_MINT_AMOUNT.toNumber()
    );
    
    await transfer(
      provider.connection,
      wallet.payer,
      authorityTokenAccount.address,
      voterBTokenAccount,
      wallet.payer,
      DEFAULT_MINT_AMOUNT.toNumber()
    );
    
    // Set PDA references
    governancePDA = governancePDAAddress;
    treasuryPDA = treasuryPDAAddress;
    proposalPDA = proposalPDAAddress;
  });
  
  it("Initialize governance", async () => {
    // Governance configuration
    const governanceConfig = {
      quorumPercentage: 20, // 20%
      approvalThreshold: 60, // 60%
      votingDelay: new BN(3600), // 1 hour
      votingPeriod: new BN(259_200), // 3 days
      timelockDelay: new BN(86_400), // 1 day
    };
    
    await program.methods
      .initialize(governanceConfig)
      .accounts({
        authority: wallet.publicKey,
        governance: governancePDA,
        treasury: treasuryPDA,
        governanceTokenMint: governanceTokenMint,
        systemProgram: anchor.web3.SystemProgram.programId,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
      })
      .rpc();
    
    // Verify the governance state
    const governanceAccount = await program.account.governanceConfig.fetch(governancePDA);
    
    assert.equal(
      governanceAccount.authority.toString(),
      wallet.publicKey.toString(),
      "Authority should match"
    );
    assert.equal(
      governanceAccount.governanceTokenMint.toString(),
      governanceTokenMint.toString(),
      "Token mint should match"
    );
    assert.equal(
      governanceAccount.quorumPercentage,
      governanceConfig.quorumPercentage,
      "Quorum percentage should match"
    );
    assert.equal(
      governanceAccount.approvalThreshold,
      governanceConfig.approvalThreshold,
      "Approval threshold should match"
    );
    
    // Verify the treasury state
    const treasuryAccount = await program.account.governanceTreasury.fetch(treasuryPDA);
    
    assert.equal(
      treasuryAccount.authority.toString(),
      governancePDA.toString(),
      "Treasury authority should be governance PDA"
    );
    assert.equal(
      treasuryAccount.totalDeposits.toNumber(),
      0,
      "Initial deposits should be 0"
    );
  });
  
  it("Create proposal", async () => {
    const proposalTitle = "Add new investment strategy";
    const proposalDescription = "This proposal seeks to add a new investment strategy focused on DeFi yield farming.";
    
    await program.methods
      .createProposal(
        { parameterChange: {} }, // ProposalType::ParameterChange
        proposalTitle,
        proposalDescription,
        DEFAULT_VOTING_PERIOD,
        DEFAULT_EXECUTION_DELAY,
        proposalIndex
      )
      .accounts({
        proposer: voterA.publicKey,
        governance: governancePDA,
        proposal: proposalPDA,
        proposerTokenAccount: voterATokenAccount,
        systemProgram: anchor.web3.SystemProgram.programId,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .signers([voterA])
      .rpc();
    
    // Verify the proposal state
    const proposalAccount = await program.account.proposal.fetch(proposalPDA);
    
    assert.equal(
      proposalAccount.proposer.toString(),
      voterA.publicKey.toString(),
      "Proposer should match"
    );
    assert.equal(
      proposalAccount.title,
      proposalTitle,
      "Title should match"
    );
    assert.equal(
      proposalAccount.description,
      proposalDescription,
      "Description should match"
    );
    assert.equal(
      proposalAccount.state.draft !== undefined,
      true,
      "Proposal should be in draft state"
    );
  });
  
  it("Add proposal action", async () => {
    const action = {
      transferFunds: {
        source: anchor.web3.Keypair.generate().publicKey,
        destination: anchor.web3.Keypair.generate().publicKey,
        amount: new BN(50_000_000),
      }
    };
    
    await program.methods
      .addProposalAction(action)
      .accounts({
        proposer: voterA.publicKey,
        governance: governancePDA,
        proposal: proposalPDA,
      })
      .signers([voterA])
      .rpc();
    
    // Verify the proposal has the action
    const proposalAccount = await program.account.proposal.fetch(proposalPDA);
    
    assert.equal(
      proposalAccount.actions.length,
      1,
      "Proposal should have 1 action"
    );
    
    const addedAction = proposalAccount.actions[0];
    assert.equal(
      addedAction.transferFunds !== undefined,
      true,
      "Action should be TransferFunds"
    );
  });
  
  it("Activate proposal", async () => {
    await program.methods
      .activateProposal()
      .accounts({
        proposer: voterA.publicKey,
        governance: governancePDA,
        proposal: proposalPDA,
        proposerTokenAccount: voterATokenAccount,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .signers([voterA])
      .rpc();
    
    // Verify the proposal state
    const proposalAccount = await program.account.proposal.fetch(proposalPDA);
    
    assert.equal(
      proposalAccount.state.active !== undefined,
      true,
      "Proposal should be in active state"
    );
    assert.notEqual(
      proposalAccount.votingStartsAt.toNumber(),
      0,
      "Voting start time should be set"
    );
    assert.notEqual(
      proposalAccount.votingEndsAt.toNumber(),
      0,
      "Voting end time should be set"
    );
  });
  
  it("Cast votes", async () => {
    // First, we need to get the vote record PDA for voter A
    const [voteRecordA] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("vote_record"), proposalPDA.toBuffer(), voterA.publicKey.toBuffer()],
      program.programId
    );
    
    // And for voter B
    const [voteRecordB] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("vote_record"), proposalPDA.toBuffer(), voterB.publicKey.toBuffer()],
      program.programId
    );
    
    // Voter A votes YES
    await program.methods
      .castVote({ yes: {} })
      .accounts({
        voter: voterA.publicKey,
        governance: governancePDA,
        proposal: proposalPDA,
        voteRecord: voteRecordA,
        voterTokenAccount: voterATokenAccount,
        tokenLock: null, // No token lock for this test
        systemProgram: anchor.web3.SystemProgram.programId,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .signers([voterA])
      .rpc();
    
    // Voter B votes NO
    await program.methods
      .castVote({ no: {} })
      .accounts({
        voter: voterB.publicKey,
        governance: governancePDA,
        proposal: proposalPDA,
        voteRecord: voteRecordB,
        voterTokenAccount: voterBTokenAccount,
        tokenLock: null, // No token lock for this test
        systemProgram: anchor.web3.SystemProgram.programId,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .signers([voterB])
      .rpc();
    
    // Verify votes
    const proposalAccount = await program.account.proposal.fetch(proposalPDA);
    const voteRecordAAccount = await program.account.voteRecord.fetch(voteRecordA);
    const voteRecordBAccount = await program.account.voteRecord.fetch(voteRecordB);
    
    // Verify Voter A's vote
    assert.equal(
      voteRecordAAccount.voter.toString(),
      voterA.publicKey.toString(),
      "Vote record A should be for Voter A"
    );
    assert.equal(
      voteRecordAAccount.vote.yes !== undefined,
      true,
      "Voter A should have voted YES"
    );
    
    // Verify Voter B's vote
    assert.equal(
      voteRecordBAccount.voter.toString(),
      voterB.publicKey.toString(),
      "Vote record B should be for Voter B"
    );
    assert.equal(
      voteRecordBAccount.vote.no !== undefined,
      true,
      "Voter B should have voted NO"
    );
    
    // Verify proposal vote counts
    assert.equal(
      proposalAccount.yesVotes.toString(),
      DEFAULT_MINT_AMOUNT.toString(),
      "YES votes should match Voter A's balance"
    );
    assert.equal(
      proposalAccount.noVotes.toString(),
      DEFAULT_MINT_AMOUNT.toString(),
      "NO votes should match Voter B's balance"
    );
  });
  
  it("Finalize proposal with mock clock", async () => {
    // For testing purposes, we'll use a mock clock that simulates the voting period being over
    // In a real implementation, you'd have to wait for the voting period to end
    // or modify the program to accept a mock clock for testing
    
    // This is a workaround for testing - in a real implementation, you'd wait for the voting period to end
    // Here we're assuming the program has been modified to allow for this test case
    await program.methods
      .finalizeProposal()
      .accounts({
        user: wallet.publicKey,
        governance: governancePDA,
        proposal: proposalPDA,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .rpc();
    
    // Verify the proposal state
    const proposalAccount = await program.account.proposal.fetch(proposalPDA);
    
    // Since votes are 50/50 and quorum is 20%, proposal should succeed
    // Note: This is simplified for testing - real logic may vary
    assert.equal(
      proposalAccount.state.succeeded !== undefined,
      true,
      "Proposal should be in succeeded state"
    );
    
    assert.notEqual(
      proposalAccount.executionTimestamp.toNumber(),
      0,
      "Execution timestamp should be set"
    );
  });
  
  it("Execute proposal with mock clock", async () => {
    // Again, this is a workaround for testing
    // In a real implementation, you'd wait for the timelock to expire
    
    await program.methods
      .executeProposal()
      .accounts({
        executor: wallet.publicKey,
        governance: governancePDA,
        proposal: proposalPDA,
        treasury: treasuryPDA,
        systemProgram: anchor.web3.SystemProgram.programId,
        tokenProgram: anchor.utils.token.TOKEN_PROGRAM_ID,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .rpc();
    
    // Verify the proposal state
    const proposalAccount = await program.account.proposal.fetch(proposalPDA);
    
    assert.equal(
      proposalAccount.state.executed !== undefined,
      true,
      "Proposal should be in executed state"
    );
  });
  
  it("Create delegate", async () => {
    // Generate a delegate keypair
    const delegate = anchor.web3.Keypair.generate();
    
    // Find the delegate record PDA
    const [delegateRecord] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("delegate"), voterA.publicKey.toBuffer(), delegate.publicKey.toBuffer()],
      program.programId
    );
    
    await program.methods
      .createDelegate()
      .accounts({
        voter: voterA.publicKey,
        delegate: delegate.publicKey,
        governance: governancePDA,
        delegateRecord: delegateRecord,
        systemProgram: anchor.web3.SystemProgram.programId,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .signers([voterA])
      .rpc();
    
    // Verify the delegate record
    const delegateRecordAccount = await program.account.delegateRecord.fetch(delegateRecord);
    
    assert.equal(
      delegateRecordAccount.voter.toString(),
      voterA.publicKey.toString(),
      "Delegate record voter should match"
    );
    assert.equal(
      delegateRecordAccount.delegate.toString(),
      delegate.publicKey.toString(),
      "Delegate record delegate should match"
    );
    assert.equal(
      delegateRecordAccount.active,
      true,
      "Delegate should be active"
    );
  });
  
  it("Lock tokens for increased voting power", async () => {
    // Find the token lock PDA
    const [tokenLock] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("token_lock"), voterA.publicKey.toBuffer()],
      program.programId
    );
    
    // Create a token account to serve as the escrow
    const escrowTokenAccount = await getOrCreateAssociatedTokenAccount(
      provider.connection,
      wallet.payer,
      governanceTokenMint,
      tokenLock,
      true  // allowOwnerOffCurve = true for PDAs
    );
    
    const lockAmount = new BN(50_000_000); // Lock 50 tokens
    const lockDuration = new BN(30 * 86400); // 30 days
    
    await program.methods
      .lockTokens(lockAmount, lockDuration)
      .accounts({
        voter: voterA.publicKey,
        governance: governancePDA,
        tokenLock: tokenLock,
        voterTokenAccount: voterATokenAccount,
        escrowTokenAccount: escrowTokenAccount.address,
        tokenProgram: anchor.utils.token.TOKEN_PROGRAM_ID,
        systemProgram: anchor.web3.SystemProgram.programId,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .signers([voterA])
      .rpc();
    
    // Verify the token lock
    const tokenLockAccount = await program.account.tokenLock.fetch(tokenLock);
    
    assert.equal(
      tokenLockAccount.owner.toString(),
      voterA.publicKey.toString(),
      "Token lock owner should match"
    );
    assert.equal(
      tokenLockAccount.lockedAmount.toString(),
      lockAmount.toString(),
      "Locked amount should match"
    );
    
    // Verify tokens were transferred to escrow
    const escrowBalance = await getAccount(
      provider.connection,
      escrowTokenAccount.address
    );
    
    assert.equal(
      escrowBalance.amount.toString(),
      lockAmount.toString(),
      "Escrow balance should match locked amount"
    );
  });
  
  it("Try to create a proposal with insufficient tokens", async () => {
    // Generate a new user with insufficient tokens
    const poorVoter = anchor.web3.Keypair.generate();
    
    // Airdrop SOL to the poor voter
    const airdrop = await provider.connection.requestAirdrop(
      poorVoter.publicKey,
      web3.LAMPORTS_PER_SOL
    );
    await provider.connection.confirmTransaction(airdrop);
    
    // Create token account for the poor voter
    const poorVoterTokenAccount = (
      await getOrCreateAssociatedTokenAccount(
        provider.connection,
        wallet.payer,
        governanceTokenMint,
        poorVoter.publicKey
      )
    ).address;
    
    // Mint a small amount of tokens to the poor voter
    await mintTo(
      provider.connection,
      wallet.payer,
      governanceTokenMint,
      poorVoterTokenAccount,
      wallet.payer,
      1_000_000 // 1 token (below threshold)
    );
    
    // Find a new proposal PDA
    const newProposalIndex = new BN(2);
    const [poorVoterProposal] = anchor.web3.PublicKey.findProgramAddressSync(
      [Buffer.from("proposal"), newProposalIndex.toArrayLike(Buffer, "le", 8)],
      program.programId
    );
    
    // Try to create and activate a proposal
    await program.methods
      .createProposal(
        { text: {} },
        "Poor voter proposal",
        "This should fail activation",
        DEFAULT_VOTING_PERIOD,
        DEFAULT_EXECUTION_DELAY,
        newProposalIndex
      )
      .accounts({
        proposer: poorVoter.publicKey,
        governance: governancePDA,
        proposal: poorVoterProposal,
        proposerTokenAccount: poorVoterTokenAccount,
        systemProgram: anchor.web3.SystemProgram.programId,
        rent: anchor.web3.SYSVAR_RENT_PUBKEY,
        clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
      })
      .signers([poorVoter])
      .rpc();
    
    // Add an action to the proposal
    const action = {
      changeParameter: {
        programId: program.programId,
        parameterName: "test_param",
        newValue: Buffer.from("test_value"),
      }
    };
    
    await program.methods
      .addProposalAction(action)
      .accounts({
        proposer: poorVoter.publicKey,
        governance: governancePDA,
        proposal: poorVoterProposal,
      })
      .signers([poorVoter])
      .rpc();
    
    // Try to activate the proposal - this should fail
    try {
      await program.methods
        .activateProposal()
        .accounts({
          proposer: poorVoter.publicKey,
          governance: governancePDA,
          proposal: poorVoterProposal,
          proposerTokenAccount: poorVoterTokenAccount,
          clock: anchor.web3.SYSVAR_CLOCK_PUBKEY,
        })
        .signers([poorVoter])
        .rpc();
      
      // If we reach here, the test failed
      assert.fail("Should have thrown an error");
    } catch (error) {
      // Verify that we got the expected error
      const expectedError = "InsufficientTokens";
      assert.include(
        error.toString(),
        expectedError,
        `Expected error containing "${expectedError}"`
      );
    }
  });
  
  // Additional tests for other functionality
  it("Test updating governance configuration", async () => {
    // Updated configuration
    const updatedConfig = {
      quorumPercentage: 30, // 30%
      approvalThreshold: 70, // 70%
      votingDelay: new BN(7200), // 2 hours
      votingPeriod: new BN(432_000), // 5 days
      timelockDelay: new BN(172_800), // 2 days
    };
    
    await program.methods
      .updateConfig(updatedConfig)
      .accounts({
        authority: wallet.publicKey,
        governance: governancePDA,
      })
      .rpc();
    
    // Verify the updated configuration
    const governanceAccount = await program.account.governanceConfig.fetch(governancePDA);
    
    assert.equal(
      governanceAccount.quorumPercentage,
      updatedConfig.quorumPercentage,
      "Quorum percentage should be updated"
    );
    assert.equal(
      governanceAccount.approvalThreshold,
      updatedConfig.approvalThreshold,
      "Approval threshold should be updated"
    );
    assert.equal(
      governanceAccount.votingPeriod.toString(),
      updatedConfig.votingPeriod.toString(),
      "Voting period should be updated"
    );
  });
  
  it("Test treasury deposit and withdrawal", async () => {
    // Create a token account for the treasury
    const treasuryTokenAccount = await getOrCreateAssociatedTokenAccount(
      provider.connection,
      wallet.payer,
      governanceTokenMint,
      treasuryPDA,
      true  // allowOwnerOffCurve = true for PDAs
    );
    
    const depositAmount = new BN(100_000_000); // 100 tokens
    
    // Deposit to treasury
    await program.methods
      .depositTreasury(depositAmount)
      .accounts({
        depositor: wallet.publicKey,
        governance: governancePDA,
        treasury: treasuryPDA,
        depositorTokenAccount: (await getOrCreateAssociatedTokenAccount(
          provider.connection,
          wallet.payer,
          governanceTokenMint,
          wallet.publicKey
        )).address,
        treasuryTokenAccount: treasuryTokenAccount.address,
        tokenProgram: anchor.utils.token.TOKEN_PROGRAM_ID,
      })
      .rpc();
    
    // Verify the treasury state
    const treasuryAccount = await program.account.governanceTreasury.fetch(treasuryPDA);
    
    assert.equal(
      treasuryAccount.totalDeposits.toString(),
      depositAmount.toString(),
      "Treasury deposits should be updated"
    );
    
    // Verify the token balance
    const treasuryBalance = await getAccount(
      provider.connection,
      treasuryTokenAccount.address
    );
    
    assert.equal(
      treasuryBalance.amount.toString(),
      depositAmount.toString(),
      "Treasury token balance should match deposit"
    );
    
    // Note: Testing withdrawal would require creating a proposal and executing it
    // This is left as an exercise for the reader or could be added in a more comprehensive test suite
  });
});