#!/bin/bash
#
# RAG Baseline Test Runner
# ========================
# Runs the stability test suite and saves baseline metrics.
#
# Usage:
#   ./scripts/run_baseline_test.sh              # Run tests
#   ./scripts/run_baseline_test.sh --save       # Run and save results
#   ./scripts/run_baseline_test.sh --docker     # Run inside Docker
#
# Environment Variables:
#   RAG_TEST_API_URL - API base URL (default: http://localhost:8000)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
API_URL="${RAG_TEST_API_URL:-http://localhost:8000}"
SAVE_RESULTS=false
USE_DOCKER=false
RESULTS_DIR="$PROJECT_ROOT/test_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --save)
            SAVE_RESULTS=true
            shift
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --api)
            API_URL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --save     Save results to test_results/ directory"
            echo "  --docker   Run tests inside Docker container"
            echo "  --api URL  Specify API base URL"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RAG STABILITY BASELINE TEST                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "📍 API URL: ${YELLOW}$API_URL${NC}"
echo -e "📁 Project: ${YELLOW}$PROJECT_ROOT${NC}"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to check API health
check_api() {
    echo -e "${BLUE}🔍 Checking API health...${NC}"
    
    if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ API is not responding at $API_URL${NC}"
        echo ""
        echo "Please ensure the API is running:"
        echo "  docker-compose up -d"
        echo "  # or"
        echo "  python3 scripts/run_api.py"
        return 1
    fi
}

# Function to run tests
run_tests() {
    echo ""
    echo -e "${BLUE}🧪 Running stability tests...${NC}"
    echo ""
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="$RESULTS_DIR/baseline_$TIMESTAMP.json"
    
    if $USE_DOCKER; then
        # Run inside Docker
        docker exec desoutter-api python3 tests/test_rag_stability.py \
            --api "$API_URL" \
            --save \
            --output "/app/test_results/baseline_$TIMESTAMP.json"
    else
        # Run locally
        cd "$PROJECT_ROOT"
        
        # Ensure Python path includes project
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        
        if $SAVE_RESULTS; then
            python3 tests/test_rag_stability.py \
                --api "$API_URL" \
                --save \
                --output "$OUTPUT_FILE"
        else
            python3 tests/test_rag_stability.py \
                --api "$API_URL"
        fi
    fi
    
    return $?
}

# Function to compare with previous baseline
compare_baseline() {
    if [ ! -d "$RESULTS_DIR" ] || [ -z "$(ls -A $RESULTS_DIR 2>/dev/null)" ]; then
        echo -e "${YELLOW}⚠️  No previous baseline found${NC}"
        return
    fi
    
    echo ""
    echo -e "${BLUE}📊 Comparing with previous baselines...${NC}"
    
    # Get the two most recent files
    LATEST=$(ls -t "$RESULTS_DIR"/baseline_*.json 2>/dev/null | head -1)
    PREVIOUS=$(ls -t "$RESULTS_DIR"/baseline_*.json 2>/dev/null | head -2 | tail -1)
    
    if [ "$LATEST" != "$PREVIOUS" ] && [ -f "$PREVIOUS" ]; then
        echo "Current:  $LATEST"
        echo "Previous: $PREVIOUS"
        
        # Extract pass rates using Python
        python3 -c "
import json
import sys

with open('$LATEST') as f:
    current = json.load(f)
    
with open('$PREVIOUS') as f:
    previous = json.load(f)

current_rate = current['summary']['pass_rate']
previous_rate = previous['summary']['pass_rate']
diff = current_rate - previous_rate

print(f'')
print(f'Pass Rate: {current_rate:.1f}% (was {previous_rate:.1f}%)')
if diff > 0:
    print(f'📈 Improved by {diff:.1f} percentage points')
elif diff < 0:
    print(f'📉 Decreased by {abs(diff):.1f} percentage points')
else:
    print(f'➡️  No change')
"
    fi
}

# Function to generate summary report
generate_report() {
    echo ""
    echo -e "${BLUE}📝 Generating baseline report...${NC}"
    
    LATEST=$(ls -t "$RESULTS_DIR"/baseline_*.json 2>/dev/null | head -1)
    
    if [ -f "$LATEST" ]; then
        python3 -c "
import json
from datetime import datetime

with open('$LATEST') as f:
    data = json.load(f)

summary = data['summary']
timing = data['timing']

print('')
print('=' * 60)
print('BASELINE REPORT')
print('=' * 60)
print(f\"Date:       {data['timestamp']}\")
print(f\"Pass Rate:  {summary['pass_rate']:.1f}%\")
print(f\"Tests:      {summary['passed']}/{summary['total_tests']} passed\")
print(f\"Avg Time:   {timing['avg_response_time_ms']:.0f}ms\")
print('')
print('By Intent:')
for intent, stats in sorted(data['by_intent'].items()):
    rate = (stats['passed'] / stats['total']) * 100
    status = '✅' if rate >= 80 else '⚠️' if rate >= 50 else '❌'
    print(f\"  {status} {intent}: {stats['passed']}/{stats['total']} ({rate:.0f}%)\")
print('')
print('Failed Tests:')
for test_id in data.get('failed_tests', [])[:5]:
    print(f\"  ❌ {test_id}\")
if len(data.get('failed_tests', [])) > 5:
    print(f\"  ... and {len(data['failed_tests']) - 5} more\")
print('=' * 60)
"
    fi
}

# Main execution
main() {
    # Check API first
    if ! check_api; then
        exit 1
    fi
    
    # Run tests
    if run_tests; then
        EXIT_CODE=0
        echo ""
        echo -e "${GREEN}✅ Tests completed successfully${NC}"
    else
        EXIT_CODE=$?
        echo ""
        echo -e "${RED}❌ Tests failed (exit code: $EXIT_CODE)${NC}"
    fi
    
    # Compare with previous if we saved results
    if $SAVE_RESULTS; then
        compare_baseline
        generate_report
    fi
    
    # Final message
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}🎉 System is STABLE - Ready for optimizations${NC}"
    else
        echo -e "${YELLOW}⚠️  Fix failing tests before proceeding${NC}"
    fi
    echo ""
    
    exit $EXIT_CODE
}

main
