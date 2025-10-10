#!/bin/bash

# Setup script for Ansible Coding Agent

echo "Setting up Ansible Coding Agent..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
ENVIRONMENT=test
VEGAS_API_KEY=your_api_key_here
VEGAS_USECASE_NAME=AnsibleCodingAgent
VEGAS_CONTEXT_NAME=AnsibleCodeContext
GIT_REPO_URL=https://github.com/your/ansible-repo.git
EOF
    echo ".env file created. Please update it with your actual credentials."
else
    echo ".env file already exists."
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your VEGAS credentials and Git repo URL"
echo "2. Run: pip install -r requirements.txt"
echo "3. Run: python agent.py"
