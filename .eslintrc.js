module.exports = {
    root: true,
    parser: '@typescript-eslint/parser',
    parserOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      project: ['./tsconfig.json', './packages/*/tsconfig.json', './sdk/tsconfig.json'],
      tsconfigRootDir: __dirname,
      ecmaFeatures: {
        jsx: true,
      },
    },
    env: {
      es2022: true,
      node: true,
      browser: true,
      jest: true,
    },
    extends: [
      'eslint:recommended',
      'plugin:@typescript-eslint/recommended',
      'plugin:@typescript-eslint/recommended-requiring-type-checking',
      'plugin:import/errors',
      'plugin:import/warnings',
      'plugin:import/typescript',
      'plugin:jest/recommended',
      'plugin:prettier/recommended',
    ],
    plugins: [
      '@typescript-eslint',
      'import',
      'jest',
      'prettier',
    ],
    settings: {
      'import/parsers': {
        '@typescript-eslint/parser': ['.ts', '.tsx', '.d.ts'],
      },
      'import/resolver': {
        typescript: {
          alwaysTryTypes: true,
          project: ['./tsconfig.json', './packages/*/tsconfig.json', './sdk/tsconfig.json'],
        },
        node: {
          extensions: ['.js', '.jsx', '.ts', '.tsx', '.d.ts'],
          moduleDirectory: ['node_modules', 'src'],
        },
      },
      'import/extensions': ['.js', '.jsx', '.ts', '.tsx', '.d.ts'],
    },
    rules: {
      // TypeScript specific rules
      '@typescript-eslint/explicit-module-boundary-types': 'warn',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': ['error', { 
        argsIgnorePattern: '^_',
        varsIgnorePattern: '^_',
        caughtErrorsIgnorePattern: '^_',
      }],
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': [
        'error',
        {
          checksVoidReturn: false,
        },
      ],
      '@typescript-eslint/naming-convention': [
        'error',
        {
          selector: 'interface',
          format: ['PascalCase'],
          prefix: ['I'],
        },
        {
          selector: 'typeAlias',
          format: ['PascalCase'],
        },
        {
          selector: 'enum',
          format: ['PascalCase'],
        },
        {
          selector: 'class',
          format: ['PascalCase'],
        },
      ],
      '@typescript-eslint/member-ordering': [
        'error',
        {
          default: [
            'public-static-field',
            'protected-static-field',
            'private-static-field',
            'public-instance-field',
            'protected-instance-field',
            'private-instance-field',
            'constructor',
            'public-static-method',
            'protected-static-method',
            'private-static-method',
            'public-instance-method',
            'protected-instance-method',
            'private-instance-method',
          ],
        },
      ],
  
      // Import rules
      'import/no-unresolved': 'error',
      'import/named': 'error',
      'import/namespace': 'error',
      'import/default': 'error',
      'import/export': 'error',
      'import/no-named-as-default': 'warn',
      'import/no-named-as-default-member': 'warn',
      'import/no-duplicates': 'error',
      'import/order': [
        'error',
        {
          'groups': [
            'builtin',
            'external',
            'internal',
            'parent',
            'sibling',
            'index',
            'object',
            'type',
          ],
          'pathGroups': [
            {
              pattern: '@minos-ai/**',
              group: 'internal',
              position: 'before',
            },
          ],
          'newlines-between': 'always',
          'alphabetize': {
            order: 'asc',
            caseInsensitive: true,
          },
        },
      ],
  
      // Best practices
      'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
      'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
      'no-alert': process.env.NODE_ENV === 'production' ? 'error' : 'warn',
      'no-param-reassign': 'error',
      'no-var': 'error',
      'prefer-const': 'error',
      'prefer-template': 'error',
      'prefer-rest-params': 'error',
      'prefer-spread': 'error',
      'prefer-destructuring': ['error', { 'object': true, 'array': false }],
      'no-duplicate-imports': 'off', // Using import/no-duplicates instead
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['../**/*/'],
              message: 'Usage of deeply relative imports is not allowed.',
            },
          ],
        },
      ],
  
      // Code style
      'curly': ['error', 'all'],
      'brace-style': ['error', '1tbs'],
      'eqeqeq': ['error', 'always', { 'null': 'ignore' }],
      'spaced-comment': ['error', 'always', { 'markers': ['/'] }],
      'no-multiple-empty-lines': ['error', { 'max': 1, 'maxEOF': 1 }],
      'no-trailing-spaces': 'error',
      'quotes': ['error', 'single', { 'avoidEscape': true }],
      'max-len': [
        'warn',
        {
          'code': 100,
          'tabWidth': 2,
          'ignoreUrls': true,
          'ignoreStrings': true,
          'ignoreTemplateLiterals': true,
          'ignoreRegExpLiterals': true,
          'ignoreComments': true,
        },
      ],
      'comma-dangle': ['error', 'always-multiline'],
      'eol-last': ['error', 'always'],
      'semi': ['error', 'always'],
      'no-empty-function': [
        'error',
        { allow: ['arrowFunctions', 'constructors'] },
      ],
  
      // Jest rules
      'jest/no-disabled-tests': 'warn',
      'jest/no-focused-tests': 'error',
      'jest/no-identical-title': 'error',
      'jest/valid-expect': 'error',
      'jest/expect-expect': 'error',
      'jest/no-test-prefixes': 'error',
      'jest/prefer-to-have-length': 'warn',
      'jest/prefer-to-be': 'warn',
      'jest/no-commented-out-tests': 'warn',
  
      // Prettier integration
      'prettier/prettier': [
        'error',
        {
          singleQuote: true,
          trailingComma: 'all',
          printWidth: 100,
          tabWidth: 2,
          semi: true,
          bracketSpacing: true,
          arrowParens: 'always',
          endOfLine: 'lf',
        },
      ],
    },
    ignorePatterns: [
      'node_modules/',
      'dist/',
      'build/',
      'coverage/',
      '.anchor/',
      'target/',
      '*.d.ts',
      '*.js.map',
      'jest.config.js',
      '.eslintrc.js',
      'commitlint.config.js',
      'babel.config.js',
      'hardhat.config.js',
      'truffle-config.js',
      'webpack.config.js',
      'next.config.js',
      'metro.config.js',
    ],
    overrides: [
      {
        files: ['**/*.test.ts', '**/*.spec.ts', '**/test/**/*.ts'],
        env: {
          jest: true,
        },
        rules: {
          '@typescript-eslint/no-explicit-any': 'off',
          '@typescript-eslint/no-non-null-assertion': 'off',
          'max-len': 'off',
        },
      },
      {
        files: ['**/*.js'],
        rules: {
          '@typescript-eslint/no-var-requires': 'off',
          '@typescript-eslint/explicit-module-boundary-types': 'off',
        },
      },
      {
        files: ['packages/contracts/**/*.ts'],
        rules: {
          '@typescript-eslint/naming-convention': 'off', // Allow snake_case for Anchor generated types
        },
      },
      {
        files: ['scripts/**/*.ts', 'scripts/**/*.js'],
        rules: {
          'no-console': 'off',
          '@typescript-eslint/no-floating-promises': 'off',
        },
      },
    ],
  };