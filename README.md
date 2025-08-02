# blueside-code

All projects rely on the uv package manager. 

## Setup
1. Create uv project.
    - 'uv init' if you have an exisiting project directory.
    - 'uv init new_app --app' to create app structure in new directory. (--lib flag is also available)
2. If project is already created run 'uv sync' to install packages defined in the lock file.
    Packages can be added and removed with 'uv add package-name' or 'uv remove package-name'.
3. Modules can be run with 'uv run module.py'