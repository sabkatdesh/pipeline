# run_full_ingest.ps1 — run from E:\pipeline
Set-StrictMode -Version Latest
Set-Location -Path 'E:\pipeline'

# Build & start services (uses E:\pipeline\.env)
docker compose up --build --detach

# Wait for API health
$apiUrl = 'http://localhost:8000'
$maxWait = 600
$waited = 0
Write-Output "Waiting up to $maxWait seconds for $apiUrl/health..."
while ($waited -lt $maxWait) {
  try {
    $r = Invoke-RestMethod -Uri "$apiUrl/health" -Method Get -TimeoutSec 5 -ErrorAction Stop
    Write-Output "API healthy: $($r | ConvertTo-Json -Depth 3)"
    break
  } catch {
    Start-Sleep -Seconds 5
    $waited += 5
    Write-Output "Waiting... ${waited}s"
  }
}
if ($waited -ge $maxWait) {
  Write-Error "Timed out waiting for API. Dumping last 200 api logs to E:\pipeline\api_logs.txt"
  docker compose logs --no-color --tail 200 api > E:\pipeline\api_logs.txt
  exit 1
}

# Trigger ingestion (empty body uses defaults in .env -> full ingestion)
Write-Output "Triggering ingestion (full run) ..."
$post = Invoke-RestMethod -Uri "$apiUrl/api/v1/ingest/run" -Method Post -Body '{}' -ContentType 'application/json' -TimeoutSec 60
$post | ConvertTo-Json -Depth 5

# Poll status every 15s up to 1800s (30 minutes)
$pollInterval = 15
$pollMax = 1800
$elapsed = 0
while ($elapsed -lt $pollMax) {
  try {
    $st = Invoke-RestMethod -Uri "$apiUrl/api/v1/ingest/status" -Method Get -TimeoutSec 60
    Write-Output "STATUS ($elapsed s): $($st.status) | fetched:$($st.papers_fetched) inserted:$($st.papers_inserted) embedded:$($st.papers_embedded)"
    if ($st.status -ne 'running') { break }
  } catch {
    Write-Output "Error polling status: $($_.Exception.Message)"
  }
  Start-Sleep -Seconds $pollInterval
  $elapsed += $pollInterval
}

# Save final status and logs
Invoke-RestMethod -Uri "$apiUrl/api/v1/ingest/status" -Method Get | ConvertTo-Json -Depth 10 > E:\pipeline\final_ingest_status.json
docker compose logs --no-color --tail 500 api > E:\pipeline\api_logs.txt
docker compose logs --no-color --tail 500 db  > E:\pipeline\db_logs.txt

# Make an LLM/RAG test call
$askBody = '{"question":"What is the transformer architecture?","top_k":3,"stream":false}'
try {
  $resp = Invoke-RestMethod -Uri "$apiUrl/api/v1/rag/ask" -Method Post -Body $askBody -ContentType 'application/json' -TimeoutSec 120
  $resp | ConvertTo-Json -Depth 10
} catch {
  Write-Error "RAG ask failed: $($_.Exception.Message)"
  docker compose logs --no-color --tail 200 api >> E:\pipeline\api_logs.txt
  exit 1
}

Write-Output "Done. Logs: E:\pipeline\api_logs.txt, E:\pipeline\db_logs.txt, final status: E:\pipeline\final_ingest_status.json"