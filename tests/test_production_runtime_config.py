from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_production_dockerfile_exists():
    dockerfile = PROJECT_ROOT / "Dockerfile.production"

    assert dockerfile.exists()


def test_production_dockerfile_uses_non_root_user():
    dockerfile = PROJECT_ROOT / "Dockerfile.production"
    content = dockerfile.read_text(encoding="utf-8")

    assert "USER appuser" in content


def test_production_dockerfile_exposes_api_port():
    dockerfile = PROJECT_ROOT / "Dockerfile.production"
    content = dockerfile.read_text(encoding="utf-8")

    assert "EXPOSE 8000" in content


def test_production_dockerfile_has_healthcheck():
    dockerfile = PROJECT_ROOT / "Dockerfile.production"
    content = dockerfile.read_text(encoding="utf-8")

    assert "HEALTHCHECK" in content
    assert "/health" in content


def test_compose_file_exists():
    compose_file = PROJECT_ROOT / "docker-compose.production.yml"

    assert compose_file.exists()


def test_compose_has_restart_policy():
    compose_file = PROJECT_ROOT / "docker-compose.production.yml"
    content = compose_file.read_text(encoding="utf-8")

    assert "restart: unless-stopped" in content


def test_compose_has_security_hardening():
    compose_file = PROJECT_ROOT / "docker-compose.production.yml"
    content = compose_file.read_text(encoding="utf-8")

    assert "no-new-privileges:true" in content
    assert "cap_drop:" in content
    assert "ALL" in content


def test_compose_has_resource_limits():
    compose_file = PROJECT_ROOT / "docker-compose.production.yml"
    content = compose_file.read_text(encoding="utf-8")

    assert "limits:" in content
    assert "memory:" in content
    assert "cpus:" in content