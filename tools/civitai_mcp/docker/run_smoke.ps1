param(
    [string]$ImageTag = "civitai-mcp-smoke"
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")).Path
$dockerfile = Join-Path $PSScriptRoot "Dockerfile.smoke"

docker build -f $dockerfile -t $ImageTag $repoRoot
docker run --rm $ImageTag
