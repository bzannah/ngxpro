-- NG FX Predictor Database Initialization Script

-- Create additional databases if needed
CREATE DATABASE IF NOT EXISTS ngfx_test_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ngfx_db TO ngfx_user;
GRANT ALL PRIVILEGES ON DATABASE ngfx_test_db TO ngfx_user;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS mlflow;
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO ngfx_user; 