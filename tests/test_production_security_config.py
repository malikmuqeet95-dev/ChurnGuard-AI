from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_TEMPLATE = PROJECT_ROOT / ".env.production.example"
CONFIG_FILE = PROJECT_ROOT / "backend" / "app" / "config.py"


def test_production_environment_template_exists():
    assert ENV_TEMPLATE.exists()


def test_production_environment_template_contains_security_settings():
    content = ENV_TEMPLATE.read_text(encoding="utf-8")

    assert "REQUIRE_API_KEY" in content
    assert "API_KEY" in content


def test_production_environment_template_does_not_contain_real_api_key():
    content = ENV_TEMPLATE.read_text(encoding="utf-8")

    assert "REQUIRE_API_KEY=false" in content
    assert "API_KEY=" in content


def test_config_file_exists():
    assert CONFIG_FILE.exists()

def test_api_key_security_module_exists():
    security_file = (
        PROJECT_ROOT
        / "backend"
        / "app"
        / "api_key_security.py"
    )

    assert security_file.exists()


def test_config_contains_api_security_settings():
    content = CONFIG_FILE.read_text(
        encoding="utf-8"
    )

    assert "require_api_key" in content
    assert "api_key" in content


def test_main_protects_prediction_endpoint():
    main_file = (
        PROJECT_ROOT
        / "backend"
        / "app"
        / "main.py"
    )

    content = main_file.read_text(
        encoding="utf-8"
    )

    assert "verify_api_key" in content
    assert "Depends(verify_api_key)" in content