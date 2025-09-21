# AWS App Runner Deployment Guide

This guide will help you deploy your Superannuation AI Advisor application to AWS App Runner.

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI installed and configured
3. Docker installed locally (for testing)
4. Git repository with your code

## Step 1: Prepare Your Repository

### 1.1 Push your code to GitHub/GitLab
```bash
git add .
git commit -m "Prepare for AWS App Runner deployment"
git push origin main
```

### 1.2 Verify your repository structure
Your repository should have:
- `backend/` directory with Python FastAPI application
- `backend/Dockerfile` (created)
- `backend/requirements.txt` (updated)
- `apprunner.yaml` (created)
- Frontend files in root directory

## Step 2: Set Up Environment Variables

### 2.1 Create a `.env` file (for local testing)
Copy `env.example` to `.env` and fill in your actual values:
```bash
cp env.example .env
```

### 2.2 Required Environment Variables
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key
- `AZURE_SPEECH_KEY`: Azure Speech Services key
- `AZURE_SPEECH_REGION`: Azure Speech Services region
- `GOOGLE_API_KEY`: Google AI (Gemini) API key
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`: Email configuration
- `NEWS_API_KEY`: News API key

## Step 3: Deploy Backend to AWS App Runner

### 3.1 Access AWS App Runner Console
1. Go to [AWS App Runner Console](https://console.aws.amazon.com/apprunner/)
2. Click "Create an App Runner service"

### 3.2 Configure Source
1. **Source Type**: Choose "Source code repository"
2. **Connect to GitHub**: Connect your GitHub account
3. **Repository**: Select your repository
4. **Branch**: Select `main` or your deployment branch
5. **Configuration file**: Choose "Use a configuration file"
6. **Configuration file path**: Enter `apprunner.yaml`

### 3.3 Configure Service
1. **Service name**: `superannuation-ai-backend`
2. **Virtual CPU**: 1 vCPU
3. **Virtual memory**: 2 GB
4. **Environment variables**: Add all your environment variables from the `.env` file

### 3.4 Configure Auto Scaling
1. **Min size**: 1
2. **Max size**: 10
3. **Concurrency**: 100

### 3.5 Review and Create
1. Review all settings
2. Click "Create & deploy"

## Step 4: Deploy Frontend to Vercel (Recommended)

### 4.1 Connect to Vercel
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository

### 4.2 Configure Build Settings
1. **Framework Preset**: Vite
2. **Root Directory**: `./` (root)
3. **Build Command**: `npm run build`
4. **Output Directory**: `dist`

### 4.3 Environment Variables
Add these environment variables in Vercel:
- `VITE_API_URL`: Your AWS App Runner backend URL (e.g., `https://your-app-runner-url.us-east-1.awsapprunner.com`)

### 4.4 Deploy
Click "Deploy" to start the deployment process.

## Alternative: Deploy Frontend to AWS S3 + CloudFront

If you prefer to keep everything on AWS:

### 4.1 Build Frontend Locally
```bash
npm run build
```

### 4.2 Create S3 Bucket
1. Go to AWS S3 Console
2. Create a new bucket with public read access
3. Upload the `dist` folder contents

### 4.3 Configure CloudFront
1. Create a CloudFront distribution
2. Set S3 bucket as origin
3. Configure custom error pages for SPA routing

### 4.4 Update API Configuration
Update your frontend to use the CloudFront URL for API calls.

## Step 5: Update Frontend API Configuration

### 5.1 Update API Base URL
In your frontend code, update the API base URL to point to your AWS App Runner backend:

```typescript
// In src/lib/supabase.ts or similar
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

### 5.2 Update CORS Settings
The backend is already configured to accept requests from your Vercel domain.

## Step 6: Test Your Deployment

### 6.1 Test Backend
1. Go to your AWS App Runner service URL
2. Visit `/health` endpoint to check if the service is running
3. Test other endpoints like `/` and `/users`

### 6.2 Test Frontend
1. Go to your Vercel deployment URL
2. Test the application functionality
3. Verify API calls are working

## Step 7: Monitor and Maintain

### 7.1 AWS App Runner Monitoring
- Monitor logs in AWS CloudWatch
- Set up alarms for errors and performance
- Monitor costs in AWS Cost Explorer

### 7.2 Vercel Monitoring
- Monitor deployments in Vercel dashboard
- Check function logs for any issues
- Monitor performance metrics

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure your frontend URL is added to the allowed origins in `backend/api.py`
   - Check that environment variables are set correctly

2. **Environment Variables Not Loading**
   - Verify all environment variables are set in AWS App Runner
   - Check the `.env` file format

3. **Build Failures**
   - Check AWS App Runner logs for specific error messages
   - Ensure all dependencies are in `requirements.txt`

4. **Database Connection Issues**
   - Verify Supabase credentials are correct
   - Check network connectivity

### Useful Commands

```bash
# Test backend locally
cd backend
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Build frontend locally
npm run build

# Test Docker build
cd backend
docker build -t superannuation-backend .
docker run -p 8000:8000 superannuation-backend
```

## Cost Optimization

1. **AWS App Runner**
   - Use appropriate instance sizes
   - Set up auto-scaling to scale down during low usage
   - Monitor costs regularly

2. **Vercel**
   - Use appropriate plan for your needs
   - Monitor bandwidth usage

## Security Considerations

1. **Environment Variables**
   - Never commit `.env` files to version control
   - Use AWS Secrets Manager for sensitive data
   - Rotate API keys regularly

2. **CORS Configuration**
   - Only allow necessary origins
   - Use HTTPS in production

3. **API Security**
   - Consider adding authentication middleware
   - Implement rate limiting
   - Use HTTPS for all communications

## Next Steps

1. Set up custom domain names
2. Implement CI/CD pipelines
3. Add monitoring and alerting
4. Implement backup strategies
5. Add security scanning

For more detailed information, refer to:
- [AWS App Runner Documentation](https://docs.aws.amazon.com/apprunner/)
- [Vercel Documentation](https://vercel.com/docs)
