import { expect } from 'chai';
import nock from 'nock';
import { MinosClient } from '../src/client';
import { AuthenticationError, APIError } from '../src/types';

describe('MinosClient', () => {
  // Constants
  const API_URL = 'https://api.minos.ai';
  const API_KEY = 'test-api-key';
  const API_SECRET = 'test-api-secret';
  
  // Test user data
  const testUser = {
    id: '9f5a9e2f-8b2a-4f9a-b8e7-c9c8b2a2a4a8',
    email: 'test@example.com',
    name: 'Test User'
  };
  
  // Test tokens
  const testTokens = {
    accessToken: 'test-access-token',
    refreshToken: 'test-refresh-token',
    expiresIn: 3600
  };

  let client: MinosClient;

  beforeEach(() => {
    // Create a new client for each test
    client = new MinosClient({
      apiUrl: API_URL,
      apiKey: API_KEY,
      apiSecret: API_SECRET
    });
    
    // Reset any previous nock interceptors
    nock.cleanAll();
  });

  afterEach(() => {
    // Ensure all nock interceptors have been used
    expect(nock.isDone()).to.be.true;
  });

  describe('Constructor', () => {
    it('should initialize with default values', () => {
      const defaultClient = new MinosClient({
        apiKey: API_KEY,
        apiSecret: API_SECRET
      });
      
      expect(defaultClient.apiUrl).to.equal('https://api.minos.ai/v1');
    });

    it('should initialize with custom values', () => {
      const customClient = new MinosClient({
        apiUrl: 'https://custom-api.minos.ai',
        apiKey: API_KEY,
        apiSecret: API_SECRET,
        timeout: 10000
      });
      
      expect(customClient.apiUrl).to.equal('https://custom-api.minos.ai');
      expect(customClient.timeout).to.equal(10000);
    });
  });

  describe('Authentication', () => {
    it('should authenticate with API key and secret', async () => {
      nock(API_URL)
        .post('/auth/api-key')
        .reply(200, {
          user: testUser,
          tokens: testTokens
        });

      await client.authenticate();
      
      expect(client.isAuthenticated).to.be.true;
      expect(client.user).to.deep.equal(testUser);
      expect(client.accessToken).to.equal(testTokens.accessToken);
    });

    it('should authenticate with email and password', async () => {
      const email = 'user@example.com';
      const password = 'password123';
      
      nock(API_URL)
        .post('/auth/login')
        .reply(200, {
          user: testUser,
          tokens: testTokens
        });

      await client.authenticateWithCredentials(email, password);
      
      expect(client.isAuthenticated).to.be.true;
      expect(client.user).to.deep.equal(testUser);
      expect(client.accessToken).to.equal(testTokens.accessToken);
    });

    it('should handle authentication errors', async () => {
      nock(API_URL)
        .post('/auth/api-key')
        .reply(401, {
          message: 'Invalid API key or secret'
        });

      try {
        await client.authenticate();
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).to.be.instanceOf(AuthenticationError);
        expect(error.message).to.equal('Invalid API key or secret');
        expect(client.isAuthenticated).to.be.false;
      }
    });

    it('should refresh the access token', async () => {
      // First authenticate to get tokens
      nock(API_URL)
        .post('/auth/api-key')
        .reply(200, {
          user: testUser,
          tokens: testTokens
        });

      await client.authenticate();
      
      // Then refresh the token
      const newTokens = {
        accessToken: 'new-access-token',
        refreshToken: 'new-refresh-token',
        expiresIn: 3600
      };
      
      nock(API_URL)
        .post('/auth/refresh')
        .reply(200, {
          tokens: newTokens
        });

      await client.refreshAccessToken();
      
      expect(client.accessToken).to.equal(newTokens.accessToken);
      expect(client.refreshToken).to.equal(newTokens.refreshToken);
    });

    it('should handle token refresh errors', async () => {
      // First authenticate to get tokens
      nock(API_URL)
        .post('/auth/api-key')
        .reply(200, {
          user: testUser,
          tokens: testTokens
        });

      await client.authenticate();
      
      // Then simulate a failed refresh
      nock(API_URL)
        .post('/auth/refresh')
        .reply(401, {
          message: 'Invalid refresh token'
        });

      try {
        await client.refreshAccessToken();
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).to.be.instanceOf(AuthenticationError);
        expect(error.message).to.equal('Invalid refresh token');
      }
    });

    it('should logout', async () => {
      // First authenticate to get tokens
      nock(API_URL)
        .post('/auth/api-key')
        .reply(200, {
          user: testUser,
          tokens: testTokens
        });

      await client.authenticate();
      
      // Then logout
      nock(API_URL)
        .post('/auth/logout')
        .reply(200, {
          success: true
        });

      await client.logout();
      
      expect(client.isAuthenticated).to.be.false;
      expect(client.accessToken).to.be.null;
      expect(client.refreshToken).to.be.null;
      expect(client.user).to.be.null;
    });
  });

  describe('API Request Handling', () => {
    beforeEach(async () => {
      // Authenticate before each test
      nock(API_URL)
        .post('/auth/api-key')
        .reply(200, {
          user: testUser,
          tokens: testTokens
        });

      await client.authenticate();
    });

    it('should make GET requests', async () => {
      const responseData = {
        id: '123',
        name: 'Test Resource'
      };
      
      nock(API_URL)
        .get('/resources/123')
        .reply(200, responseData);

      const response = await client.request('GET', '/resources/123');
      
      expect(response).to.deep.equal(responseData);
    });

    it('should make POST requests', async () => {
      const requestData = {
        name: 'New Resource'
      };
      
      const responseData = {
        id: '456',
        name: 'New Resource',
        createdAt: '2025-05-17T12:00:00Z'
      };
      
      nock(API_URL)
        .post('/resources', requestData)
        .reply(201, responseData);

      const response = await client.request('POST', '/resources', requestData);
      
      expect(response).to.deep.equal(responseData);
    });

    it('should make PUT requests', async () => {
      const requestData = {
        name: 'Updated Resource'
      };
      
      const responseData = {
        id: '789',
        name: 'Updated Resource',
        updatedAt: '2025-05-17T12:00:00Z'
      };
      
      nock(API_URL)
        .put('/resources/789', requestData)
        .reply(200, responseData);

      const response = await client.request('PUT', '/resources/789', requestData);
      
      expect(response).to.deep.equal(responseData);
    });

    it('should make DELETE requests', async () => {
      nock(API_URL)
        .delete('/resources/999')
        .reply(204);

      const response = await client.request('DELETE', '/resources/999');
      
      expect(response).to.be.undefined;
    });

    it('should handle request errors', async () => {
      nock(API_URL)
        .get('/resources/nonexistent')
        .reply(404, {
          message: 'Resource not found'
        });

      try {
        await client.request('GET', '/resources/nonexistent');
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).to.be.instanceOf(APIError);
        expect(error.message).to.equal('Resource not found');
        expect(error.status).to.equal(404);
      }
    });

    it('should handle network errors', async () => {
      nock(API_URL)
        .get('/resources/network-error')
        .replyWithError('Network error');

      try {
        await client.request('GET', '/resources/network-error');
        expect.fail('Should have thrown an error');
      } catch (error) {
        expect(error).to.be.instanceOf(Error);
        expect(error.message).to.include('Network error');
      }
    });

    it('should handle authentication token expiration', async () => {
      // First request fails with 401
      nock(API_URL)
        .get('/resources/protected')
        .reply(401, {
          message: 'Access token expired'
        });
      
      // Token refresh succeeds
      const newTokens = {
        accessToken: 'new-access-token',
        refreshToken: 'new-refresh-token',
        expiresIn: 3600
      };
      
      nock(API_URL)
        .post('/auth/refresh')
        .reply(200, {
          tokens: newTokens
        });
      
      // Retry with new token succeeds
      const responseData = {
        id: 'protected',
        name: 'Protected Resource'
      };
      
      nock(API_URL)
        .get('/resources/protected')
        .reply(200, responseData);

      const response = await client.request('GET', '/resources/protected', null, true);
      
      expect(response).to.deep.equal(responseData);
      expect(client.accessToken).to.equal(newTokens.accessToken);
    });

    it('should respect request timeout', async () => {
      const timeoutClient = new MinosClient({
        apiUrl: API_URL,
        apiKey: API_KEY,
        apiSecret: API_SECRET,
        timeout: 1000 // 1 second timeout
      });
      
      // Authenticate first
      nock(API_URL)
        .post('/auth/api-key')
        .reply(200, {
          user: testUser,
          tokens: testTokens
        });

      await timeoutClient.authenticate();
      
      // Request that will timeout
      nock(API_URL)
        .get('/resources/slow')
        .delay(2000) // 2 seconds delay (longer than timeout)
        .reply(200, {});

      try {
        await timeoutClient.request('GET', '/resources/slow');
        expect.fail('Should have thrown a timeout error');
      } catch (error) {
        expect(error.message).to.include('timeout');
      }
    });
  });

  describe('Utility Methods', () => {
    it('should provide version information', () => {
      const version = client.getVersion();
      
      expect(version).to.be.a('string');
      expect(version).to.match(/^\d+\.\d+\.\d+$/); // Semantic version format
    });

    it('should build correct query URLs', () => {
      const params = {
        limit: 10,
        offset: 20,
        filter: 'active',
        sort: 'name'
      };
      
      const url = client.buildUrl('/resources', params);
      
      // URL should include all parameters
      expect(url).to.equal('/resources?limit=10&offset=20&filter=active&sort=name');
    });

    it('should handle empty query parameters', () => {
      const url = client.buildUrl('/resources', {});
      
      expect(url).to.equal('/resources');
    });

    it('should encode query parameters correctly', () => {
      const params = {
        q: 'search term with spaces',
        tags: ['tag1', 'tag2'],
        filter: 'name=John Doe&age>30'
      };
      
      const url = client.buildUrl('/resources', params);
      
      expect(url).to.include('q=search%20term%20with%20spaces');
      expect(url).to.include('tags=tag1%2Ctag2');
      expect(url).to.include('filter=name%3DJohn%20Doe%26age%3E30');
    });
  });

  describe('Webhook Validation', () => {
    it('should validate webhook signatures', () => {
      const webhookSecret = 'webhook-secret';
      const payload = JSON.stringify({ event: 'test', data: { id: '123' } });
      const timestamp = Math.floor(Date.now() / 1000).toString();
      
      // Generate signature
      const signature = client.generateWebhookSignature(webhookSecret, payload, timestamp);
      
      // Validate signature
      const isValid = client.validateWebhookSignature(
        webhookSecret,
        payload,
        timestamp,
        signature
      );
      
      expect(isValid).to.be.true;
    });

    it('should reject invalid webhook signatures', () => {
      const webhookSecret = 'webhook-secret';
      const payload = JSON.stringify({ event: 'test', data: { id: '123' } });
      const timestamp = Math.floor(Date.now() / 1000).toString();
      
      // Generate signature
      const signature = client.generateWebhookSignature(webhookSecret, payload, timestamp);
      
      // Validate with wrong secret
      const isValid = client.validateWebhookSignature(
        'wrong-secret',
        payload,
        timestamp,
        signature
      );
      
      expect(isValid).to.be.false;
    });

    it('should reject expired webhook signatures', () => {
      const webhookSecret = 'webhook-secret';
      const payload = JSON.stringify({ event: 'test', data: { id: '123' } });
      
      // Timestamp from 10 minutes ago (beyond the 5-minute tolerance)
      const timestamp = (Math.floor(Date.now() / 1000) - 600).toString();
      
      // Generate signature
      const signature = client.generateWebhookSignature(webhookSecret, payload, timestamp);
      
      // Validate with expired timestamp
      const isValid = client.validateWebhookSignature(
        webhookSecret,
        payload,
        timestamp,
        signature
      );
      
      expect(isValid).to.be.false;
    });
  });
});