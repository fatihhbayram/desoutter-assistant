# 🖥️ Proxmox Deployment Guide

Production deployment of Desoutter Assistant on Proxmox VE infrastructure with GPU passthrough.

---

## 📋 Table of Contents

- [Infrastructure Overview](#infrastructure-overview)
- [Prerequisites](#prerequisites)
- [VM Configuration](#vm-configuration)
- [GPU Passthrough Setup](#gpu-passthrough-setup)
- [Docker Installation](#docker-installation)
- [Network Configuration](#network-configuration)
- [Deployment Steps](#deployment-steps)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Backup & Recovery](#backup--recovery)

---

## 🏗️ Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      PROXMOX VE HOST                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              VM: ai.server (Ubuntu 24.04)               │   │
│  │              GPU Passthrough: RTX A2000 6GB             │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Ollama    │  │  ChromaDB   │  │  Desoutter  │     │   │
│  │  │   :11434    │  │   :8000     │  │    API      │     │   │
│  │  │   (GPU)     │  │             │  │   :8000     │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   MongoDB   │  │    n8n      │  │  Frontend   │     │   │
│  │  │   :27017    │  │   :5678     │  │   :3001     │     │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  │                                                         │   │
│  │              Docker Network: ai-net                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                    Network: 192.168.1.x                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 100 GB SSD | 500 GB NVMe |
| **GPU** | NVIDIA GTX 1060 6GB | NVIDIA RTX A2000/3060+ |
| **Network** | 1 Gbps | 1 Gbps |

### Software Requirements

| Software | Version |
|----------|---------|
| Proxmox VE | 8.0+ |
| Ubuntu (VM) | 24.04 LTS |
| Docker | 24.0+ |
| Docker Compose | 2.20+ |
| NVIDIA Driver | 535+ |
| NVIDIA Container Toolkit | Latest |

---

## 🖥️ VM Configuration

### Proxmox VM Settings

Create a new VM with these specifications:

```
# General
VM ID: 100
Name: ai.server
Start at boot: Yes

# OS
ISO: ubuntu-24.04-server-amd64.iso
Type: Linux
Version: 6.x - 2.6 Kernel

# System
Machine: q35
BIOS: OVMF (UEFI)
EFI Storage: local-lvm
Add TPM: No

# Disks
Bus/Device: VirtIO Block
Storage: local-lvm
Disk size: 200 GB
SSD emulation: Yes
Discard: Yes

# CPU
Sockets: 1
Cores: 8
Type: host

# Memory
Memory: 32768 MB
Minimum memory: 16384 MB
Ballooning: Yes

# Network
Bridge: vmbr0
Model: VirtIO
VLAN Tag: (none)
Firewall: Yes
```

### VM Configuration File

Edit `/etc/pve/qemu-server/100.conf`:

```ini
agent: 1
balloon: 16384
boot: order=scsi0;net0
cores: 8
cpu: host
hostpci0: 0000:01:00,pcie=1,x-vga=0
machine: q35
memory: 32768
meta: creation-qemu=8.1.2,ctime=1700000000
name: ai.server
net0: virtio=BC:24:11:XX:XX:XX,bridge=vmbr0,firewall=1
numa: 0
onboot: 1
ostype: l26
scsi0: local-lvm:vm-100-disk-0,discard=on,size=200G,ssd=1
scsihw: virtio-scsi-single
smbios1: uuid=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
sockets: 1
vmgenid: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

---

## 🎮 GPU Passthrough Setup

### Step 1: Enable IOMMU on Proxmox Host

Edit `/etc/default/grub`:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet intel_iommu=on iommu=pt"
# For AMD: amd_iommu=on
```

Update GRUB and reboot:

```bash
update-grub
reboot
```

### Step 2: Load VFIO Modules

Edit `/etc/modules`:

```
vfio
vfio_iommu_type1
vfio_pci
vfio_virqfd
```

### Step 3: Blacklist NVIDIA Drivers on Host

Create `/etc/modprobe.d/blacklist-nvidia.conf`:

```
blacklist nouveau
blacklist nvidia
blacklist nvidiafb
```

### Step 4: Bind GPU to VFIO

Find your GPU IDs:

```bash
lspci -nn | grep -i nvidia
# Example output: 01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA107 [10de:25a2] (rev a1)
```

Create `/etc/modprobe.d/vfio.conf`:

```
options vfio-pci ids=10de:25a2,10de:2291  # GPU and Audio
```

Update initramfs and reboot:

```bash
update-initramfs -u
reboot
```

### Step 5: Install NVIDIA Drivers in VM

After VM is created and Ubuntu installed:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Reboot
sudo reboot

# Verify GPU
nvidia-smi
```

Expected output:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xxx       Driver Version: 535.xxx       CUDA Version: 12.x  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A2000    Off  | 00000000:01:00.0 Off |                  Off |
| 30%   35C    P8     8W /  70W |      0MiB /  6144MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

## 🐳 Docker Installation

### Install Docker Engine

```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

### Install NVIDIA Container Toolkit

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

---

## 🌐 Network Configuration

### Create Docker Network

```bash
# Create bridge network for AI services
docker network create ai-net
```

### Firewall Rules (UFW)

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow frontend
sudo ufw allow 3001/tcp

# Allow API
sudo ufw allow 8000/tcp

# Enable firewall
sudo ufw enable
```

---

## 🚀 Deployment Steps

### Step 1: Clone Repository

```bash
cd ~
git clone https://github.com/fatihhbayram/desoutter-assistant.git
cd desoutter-assistant
```

### Step 2: Create Environment File

```bash
cp .env.example .env
nano .env
```

Configure for Proxmox environment:

```bash
# MongoDB (container name in ai-net)
MONGO_HOST=mongodb
MONGO_PORT=27017
MONGO_DATABASE=desoutter

# Ollama (container name in ai-net)
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
OLLAMA_KEEP_ALIVE=24h

# GPU Acceleration
EMBEDDING_DEVICE=cuda

# Production JWT Secret
JWT_SECRET=$(openssl rand -hex 32)
```

### Step 3: Create Data Directories

```bash
mkdir -p data/logs data/exports data/documents/manuals data/documents/bulletins
```

### Step 4: Start Services

```bash
# Start all services
docker compose -f docker-compose.desoutter.yml up -d

# Verify services
docker compose ps

# Check logs
docker compose logs -f desoutter-api
```

### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test login
curl -X POST http://localhost:8000/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}'
```

---

## 📊 Monitoring & Maintenance

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# GPU status
nvidia-smi

# Docker container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Docker resource usage
docker stats --no-stream
```

### Log Locations

| Service | Log Command |
|---------|-------------|
| API | `docker logs -f desoutter-api` |
| Frontend | `docker logs -f desoutter-frontend` |
| MongoDB | `docker logs -f mongodb` |
| Ollama | `docker logs -f ollama` |

### Common Operations

```bash
# Restart API after config change
docker restart desoutter-api

# Update container image
docker compose pull
docker compose up -d

# Clear response cache
docker exec desoutter-api python3 -c "from src.llm.rag_engine import RAGEngine; r=RAGEngine(); r.clear_cache()"

# Re-ingest documents
docker exec desoutter-api python3 scripts/ingest_documents.py
```

### Systemd Service (Optional)

Create `/etc/systemd/system/desoutter.service`:

```ini
[Unit]
Description=Desoutter Assistant
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/adentechio/desoutter-assistant
ExecStart=/usr/bin/docker compose -f docker-compose.desoutter.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose.desoutter.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable desoutter
sudo systemctl start desoutter
```

---

## 💾 Backup & Recovery

### Backup Strategy

| Data | Location | Backup Method |
|------|----------|---------------|
| MongoDB | Docker volume | `mongodump` |
| ChromaDB | Docker volume | Volume snapshot |
| Documents | `/app/documents` | rsync |
| Config | `.env`, `config/` | Git + encrypted storage |

### MongoDB Backup

```bash
# Create backup
docker exec mongodb mongodump --out /backup/$(date +%Y%m%d)

# Copy to host
docker cp mongodb:/backup ./backups/

# Restore from backup
docker exec mongodb mongorestore /backup/20250104
```

### Proxmox VM Backup

```bash
# From Proxmox host
vzdump 100 --storage local --mode snapshot --compress zstd

# Schedule in Proxmox UI:
# Datacenter → Backup → Add
# Schedule: daily at 02:00
# Retention: 7 days
```

### Disaster Recovery

1. Restore Proxmox VM from backup
2. Start VM and verify GPU passthrough
3. Start Docker services: `docker compose up -d`
4. Restore MongoDB: `mongorestore /backup/latest`
5. Verify system: `curl http://localhost:8000/health`

---

## 🔒 Security Considerations

- [ ] Change default passwords (admin123, tech123)
- [ ] Use strong JWT_SECRET (32+ random characters)
- [ ] Enable HTTPS via reverse proxy (Nginx/Caddy)
- [ ] Restrict CORS to frontend domain
- [ ] Enable Proxmox firewall rules
- [ ] Regular security updates: `apt update && apt upgrade`
- [ ] Monitor access logs

---

## 📚 Additional Resources

- [Proxmox VE Documentation](https://pve.proxmox.com/pve-docs/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Documentation](https://docs.docker.com/)
- [Ollama GPU Setup](https://ollama.ai/docs/gpu)
