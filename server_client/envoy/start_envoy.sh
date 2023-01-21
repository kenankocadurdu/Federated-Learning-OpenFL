#!/bin/bash
set -e

fx envoy start -n client_1 --disable-tls --envoy-config-path envoy_config.yaml -dh localhost -dp 50051
