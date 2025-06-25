import tomllib
from pathlib import Path

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

project_deps = data.get("project", {}).get("dependencies", [])
dev_deps = data.get("project", {}).get("optional-dependencies", {}).get("dev", [])

# Combine and deduplicate
all_deps = sorted(set(project_deps + dev_deps))

# Output to requirements.txt
Path("requirements.txt").write_text("\n".join(all_deps) + "\n")

print("requirements.txt generated from pyproject.toml.")