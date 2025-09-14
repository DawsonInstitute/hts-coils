# MyBinder Configuration

This directory contains configuration files for deploying HTS coils notebooks on MyBinder.org.

## Files

- `runtime.txt` - Specifies Python 3.11 for reproducible environment
- `requirements.txt` - Lightweight dependencies for MyBinder (created from main requirements)
- `postBuild` - Post-installation script for Jupyter extensions (if needed)

## Usage

These files enable one-click deployment to MyBinder.org, allowing anyone to run the HTS coil notebooks interactively without installing dependencies locally.

## Resource Constraints

MyBinder has the following limitations:
- 1-2GB RAM limit
- 10-minute inactivity timeout  
- 6-hour maximum runtime
- ~1Mbit outgoing bandwidth
- No persistent storage

The configuration has been optimized to work within these constraints.