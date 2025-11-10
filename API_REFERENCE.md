# 📡 REST API Reference

Complete API documentation for Financial Analyst Advisor.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required. In production, add:
- API Key authentication
- OAuth2
- JWT tokens

## Response Format

All responses are JSON with the following structure:

```json
{
  "status": "success|error",
  "data": {},
  "timestamp": "2024-11-09T10:30:00",
  "error": null
}
```

---

## 🏥 General Endpoints

### Health Check

Check API health and component status.

**Endpoint**: `GET /health`

**Query Parameters**: None

**Response**:
```json
{
  "status": "healthy",
  "rag_pipeline_loaded": true,
  "model_loaded": true,
  "timestamp": "2024-11-09T10:30:00"
}
```

**Example**:
```bash
curl http://localhost:8000/health
```

---

### Get Statistics

Retrieve system statistics.

**Endpoint**: `GET /api/v1/stats`

**Query Parameters**: None

**Response**:
```json
{
  "rag_pipeline": {
    "persist_dir": "data/vector_db",
    "collection_count": 1250
  },
  "timestamp": "2024-11-09T10:30:00"
}
```

**Example**:
```bash
curl http://localhost:8000/api/v1/stats
```

---

## 🔍 Search Endpoints

### Search Financial Documents

Search vector database for relevant documents using semantic similarity.

**Endpoint**: `POST /api/v1/search`

**Request Body**:
```json
{
  "query": "What was the total revenue for 2024?",
  "k": 5,
  "score_threshold": 0.3
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Search query (5-500 chars) |
| k | integer | No | Number of results (1-20, default: 5) |
| score_threshold | float | No | Min similarity (0.0-1.0, default: 0.3) |

**Response**:
```json
{
  "query": "What was the total revenue for 2024?",
  "results": [
    {
      "content": "Apple Inc. reported total revenue of $383.3 billion...",
      "metadata": {
        "source": "data/raw_reports/AAPL_10K.txt",
        "filename": "AAPL_10K.txt",
        "cik": "0000320193",
        "filing_type": "10-K",
        "chunk_index": 5,
        "total_chunks": 23
      },
      "similarity_score": 0.87
    },
    {
      "content": "Revenue increased 5% year-over-year...",
      "metadata": { ... },
      "similarity_score": 0.82
    }
  ],
  "total_results": 2,
  "timestamp": "2024-11-09T10:30:00"
}
```

**Status Codes**:
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `503 Service Unavailable` - RAG pipeline not loaded

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "revenue growth",
    "k": 3,
    "score_threshold": 0.5
  }'
```

---

## 💡 Analysis Endpoints

### Analyze Financial Question

Analyze a financial question using RAG + fine-tuned model.

**Endpoint**: `POST /api/v1/analyze`

**Request Body**:
```json
{
  "question": "What are the main business risks?",
  "company_filter": "0000320193",
  "include_context": true
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| question | string | Yes | Financial question (5-500 chars) |
| company_filter | string | No | Filter by company CIK |
| include_context | boolean | No | Include retrieved context (default: true) |

**Response**:
```json
{
  "question": "What are the main business risks?",
  "answer": "According to Apple's 10-K filing, the main risks include supply chain disruptions, market competition, regulatory challenges, and currency fluctuations. The company emphasizes its reliance on key suppliers and the importance of maintaining retail relationships.",
  "confidence": 0.85,
  "sources": [
    {
      "content": "Risk Factors section mentioning supply chain...",
      "metadata": {
        "filing_type": "10-K",
        "cik": "0000320193"
      },
      "similarity_score": 0.92
    }
  ],
  "retrieved_context": "RETRIEVED FINANCIAL DOCUMENTS:\n\n[Document 1]\nFiling: 10-K\n...",
  "timestamp": "2024-11-09T10:30:00"
}
```

**Status Codes**:
- `200 OK` - Success
- `400 Bad Request` - Invalid question
- `503 Service Unavailable` - Model not loaded

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How did revenue change compared to last year?",
    "include_context": true
  }'
```

---

## 📤 Data Management Endpoints

### Ingest Documents

Add new financial documents to vector database.

**Endpoint**: `POST /api/v1/ingest`

**Request Body**:
```json
{
  "file_paths": [
    "data/raw_reports/AAPL_10K.txt",
    "data/raw_reports/MSFT_10Q.txt"
  ],
  "force_reprocess": false
}
```

**Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file_paths | array | Yes | List of file paths to ingest |
| force_reprocess | boolean | No | Force reprocessing (default: false) |

**Response**:
```json
{
  "status": "success",
  "chunks_added": 245,
  "files_processed": 2,
  "timestamp": "2024-11-09T10:30:00"
}
```

**Status Codes**:
- `200 OK` - Success
- `400 Bad Request` - No valid files
- `503 Service Unavailable` - RAG pipeline not loaded

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["data/raw_reports/TSLA_10K.txt"]
  }'
```

---

## 🤖 Generation Endpoints

### Generate Text

Generate text using fine-tuned model.

**Endpoint**: `POST /api/v1/generate`

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| prompt | string | Yes | Input prompt (5-1000 chars) |
| max_tokens | integer | No | Max tokens (1-1000, default: 256) |
| temperature | float | No | Temperature (0.1-2.0, default: 0.7) |

**Response**:
```json
{
  "prompt": "Financial analysis is important because",
  "generated_text": "Financial analysis is important because it provides insight into a company's financial health, performance trends, and future prospects. By analyzing financial statements, investors can make informed decisions about investment opportunities. Key metrics like revenue growth, profit margins, and cash flow are essential indicators of business viability.",
  "timestamp": "2024-11-09T10:30:00"
}
```

**Status Codes**:
- `200 OK` - Success
- `422 Unprocessable Entity` - Invalid parameters
- `503 Service Unavailable` - Model not loaded

**Example**:
```bash
curl -X POST "http://localhost:8000/api/v1/generate?prompt=Explain%20financial%20risk&max_tokens=200&temperature=0.7"
```

---

## 🔧 Advanced Usage

### Combining Search and Analysis

```bash
# 1. Search for relevant documents
RESULTS=$(curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "revenue trends", "k": 3}')

# 2. Ask follow-up question based on results
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the year-over-year growth rate?"
  }'
```

### Batch Analysis

```bash
#!/bin/bash

# Analyze multiple questions
questions=(
  "What was the revenue?"
  "What are the risks?"
  "What is the strategy?"
)

for q in "${questions[@]}"; do
  echo "Analyzing: $q"
  curl -X POST http://localhost:8000/api/v1/analyze \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$q\"}" | jq '.'
done
```

---

## 📊 Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing the issue",
  "status_code": 400
}
```

### Common Errors

| Status | Error | Cause | Solution |
|--------|-------|-------|----------|
| 400 | Bad Request | Invalid parameters | Check parameter types/ranges |
| 422 | Unprocessable Entity | Invalid field values | Validate field requirements |
| 503 | Service Unavailable | Component not loaded | Restart API server |
| 500 | Internal Error | Unexpected error | Check server logs |

**Example Error**:
```json
{
  "detail": "Search error: query string too short (minimum 5 characters)",
  "status_code": 400
}
```

---

## 🔐 Rate Limiting (Future)

Future releases will include rate limiting:
- 100 requests/minute for search
- 50 requests/minute for analysis
- 10 requests/minute for model generation

---

## 📈 Performance Guidelines

### Response Time Expectations

| Endpoint | Typical Time | Max Time |
|----------|-------------|----------|
| /health | <10ms | 100ms |
| /api/v1/search | 100-500ms | 2s |
| /api/v1/analyze | 1-3s | 10s |
| /api/v1/generate | 5-15s | 30s |
| /api/v1/ingest | 1-5s/file | 60s |

### Optimization Tips

1. **Search**: Reduce `k` parameter or increase `score_threshold`
2. **Analysis**: Use `include_context=false` for faster responses
3. **Generation**: Reduce `max_tokens` for faster generation
4. **Batch Operations**: Use background processing for large ingestions

---

## 🔄 Async Endpoints (Future)

Planned async endpoints for long-running operations:

```bash
# Start async job
POST /api/v1/async/train
POST /api/v1/async/ingest

# Check job status
GET /api/v1/async/jobs/{job_id}

# Get results
GET /api/v1/async/jobs/{job_id}/results
```

---

## 📚 SDK Examples

### Python Client

```python
from api_client import FinancialAnalystClient

client = FinancialAnalystClient("http://localhost:8000")

# Search
results = client.search("revenue growth", k=3)

# Analyze
response = client.analyze("What are the risks?")

# Generate
text = client.generate("Financial analysis is...")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8000';

async function search(query) {
  const response = await axios.post(
    `${API_URL}/api/v1/search`,
    { query, k: 5 }
  );
  return response.data;
}

async function analyze(question) {
  const response = await axios.post(
    `${API_URL}/api/v1/analyze`,
    { question }
  );
  return response.data;
}
```

### cURL Cheatsheet

```bash
# Health check
curl http://localhost:8000/health

# Search (POST)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"revenue","k":5}'

# Analyze (POST)
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"question":"What were the main risks?"}'

# Generate (GET with params)
curl "http://localhost:8000/api/v1/generate?prompt=Explain%20revenue&max_tokens=100"

# Pretty print JSON
curl -s http://localhost:8000/health | jq .
```

---

## 📝 API Versioning

Current version: **v1**

- Base path: `/api/v1/`
- Version in URL ensures backward compatibility
- Future versions: `/api/v2/`, etc.

---

## 🧪 Testing Endpoints

### Swagger UI (Interactive)

Visit: `http://localhost:8000/docs`

- Try endpoints interactively
- View request/response schemas
- Test with different parameters

### ReDoc (Read-only)

Visit: `http://localhost:8000/redoc`

- Clean API documentation
- Better for reading

---

## 🚀 Deployment Considerations

### CORS Headers (Set in Production)

```python
# In app.py - configure for your domain
CORSMiddleware(
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

### Security Headers

Add in production:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
```

### API Gateway Integration

```
Client → API Gateway → Load Balancer → API Instances
                    ↓
            Rate Limiting
            Authentication
            Request Logging
```

---

## 📞 Support

- **Interactive Docs**: http://localhost:8000/docs
- **Python Client**: `python api_client.py`
- **Example Usage**: `python examples.py`
- **Issues**: Check error responses and logs

---

**API Version**: 1.0.0
**Last Updated**: November 2024
**Status**: Production Ready ✅
