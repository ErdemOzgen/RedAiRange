# Red AI Range (RAR)

![](./frontend/public/redai.png)

## Overview

Red AI Range (RAR) is a comprehensive security platform designed specifically for AI red teaming and vulnerability assessment. It creates realistic environments where security professionals can systematically discover, analyze, and mitigate AI vulnerabilities through controlled testing scenarios.

As organizations increasingly integrate AI systems into critical infrastructure, the need for robust security testing has become essential. RAR addresses this need by providing a standardized framework that consolidates various AI vulnerabilities in one accessible environment for both academic research and industrial security operations.

## Core Capabilities

### Red Team Operations Support

RAR serves as a complete operational platform for red teams focused on AI security. It enables teams to:

- Conduct systematic assessments of AI system security boundaries
- Simulate sophisticated attack scenarios against production-like environments
- Document and report findings in a standardized format
- Train new team members on AI-specific attack vectors
- Develop custom testing methodologies for proprietary AI systems
- Validate the effectiveness of security controls and mitigations

### Containerized Architecture

Due to the inherent complexity of AI libraries and their dependencies, RAR implements a sophisticated Docker-based architecture. This approach resolves the nearly impossible task of installing multiple, often conflicting AI frameworks in a single environment. The containerized structure:

- Isolates conflicting dependencies between different AI frameworks
- Maintains consistent testing environments across deployments
- Enables rapid reset of testing environments to baseline configurations
- Supports parallel testing of multiple vulnerability types

### Advanced Stack Management System

RAR features a specialized stack management system that streamlines the preparation and deployment of Docker containers. The system includes dedicated configuration areas for:

- Vulnerable AI system deployment specifications
- Testing tool configuration and deployment parameters
- Automatic generation of docker-compose files with appropriate networking
- Environment variable management for secure credential handling

Each configuration area guides users through the proper setup of either vulnerability targets or testing tools, ensuring consistent deployments.

### Intuitive Deployment Controls

The platform offers streamlined deployment options through a user-friendly interface:

- **Arsenal Button**: Deploys containers with security testing tools, vulnerability scanners, and exploitation frameworks necessary for comprehensive AI security assessment (appends "_arsenal" to stack name for clear identification)
- **Target Button**: Deploys containers with intentionally vulnerable AI systems configured for specific testing scenarios (appends "_ai_target" to stack name)
- **Compose Button**: Creates test stacks without name modifications for development and customization purposes

### Remote Agent Architecture

RAR includes a sophisticated agent system that enables connections to remote RAR installations using secure authentication mechanisms. This distributed architecture allows security teams to:

- Leverage specialized hardware resources (such as GPU clusters for testing compute-intensive AI vulnerabilities)
- Coordinate testing activities across geographically distributed teams
- Centralize control of multiple testing environments from a single management console
- Deploy specialized testing environments on cloud platforms when needed

For example, teams can connect to an AWS-hosted RAR instance with GPU resources to test vulnerabilities in large language models that require significant computational power, all controlled through a unified interface.

### Comprehensive Recording Capabilities

The platform includes built-in session recording functionality that supports knowledge transfer and documentation requirements:

- High-quality video capture of testing sessions
- Timestamped activity logging for detailed review
- Secure storage and download options for training materials
- Customizable recording parameters for different documentation needs

This feature is particularly valuable for creating training materials and maintaining documentation of vulnerability demonstrations for stakeholders.

### Docker-in-Docker Implementation

RAR operates as a Docker container while controlling other Docker containers through the Docker socket mounted as a volume. This sophisticated architecture provides numerous advantages:

- **Enhanced Isolation**: Each testing component operates in a strictly controlled environment
- **Precise Resource Management**: Computing resources can be allocated specifically to the needs of individual test scenarios
- **Efficient Cleanup**: Testing environments can be destroyed without residual artifacts
- **Version Control**: Specific versions of AI frameworks can be maintained for consistent vulnerability reproduction
- **Parallel Operations**: Multiple testing scenarios can be executed simultaneously without interference
- **Simplified Deployment**: Complex installation procedures are encapsulated within container definitions
- **Cross-Platform Consistency**: Testing environments remain consistent regardless of the host system

## Applications and Use Cases

RAR serves as a centralized platform for security professionals across various domains:

### For Security Researchers
- Systematic exploration of novel AI attack vectors
- Controlled testing of theoretical vulnerabilities
- Development of new detection and mitigation techniques
- Publication of reproducible security findings

### For Corporate Security Teams
- Validation of AI system security before production deployment
- Regular security assessments of deployed AI systems
- Training and skill development for security personnel
- Demonstration of security capabilities to stakeholders

### For Educational Institutions
- Hands-on training in AI security principles
- Practical workshops on emerging AI threats
- Research into fundamental AI security challenges
- Development of standardized AI security curricula

## Getting Started

[Detailed installation and usage instructions would be included here in the actual README]

## Contributing

[Information about how to contribute to the project would be included here]