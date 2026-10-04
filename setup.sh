# setup.sh
# Script for installing and/or activating a virtualenv. Source it, don't execute it:
#
#     source setup.sh


##-----------------------------------------------------------------------------
## pre-setup helpers, don't touch
##-----------------------------------------------------------------------------

# BASH_SOURCE is empty under zsh; zsh sets $0 to the script path for a sourced file.
path_of_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )" && pwd )"

add_to_python_path()
{
    export PYTHONPATH=$1${PYTHONPATH:+:${PYTHONPATH}}
    echo "Added $1 to your PYTHONPATH."
}

add_to_path()
{
    export PATH=$1${PATH:+:${PATH}}
    echo "Added $1 to your PATH."
}


##-----------------------------------------------------------------------------
## setup paths
##-----------------------------------------------------------------------------

export PROJECT_PATH=${path_of_this_dir}

add_to_path ${PROJECT_PATH}/scripts
add_to_python_path ${PROJECT_PATH}/python


##-----------------------------------------------------------------------------
## setup virtualenv
##-----------------------------------------------------------------------------

venv_name="${PROJECT_PATH}/.venv"

KERNEL_NAME="ioi-hydra"

if [ -f ${venv_name}/bin/activate ]; then
    source ${venv_name}/bin/activate
else
    echo "Setting up virtualenv ${venv_name}"
    python3 -m venv ${venv_name}
    source ${venv_name}/bin/activate
    pip install --upgrade pip
    pip install -r "${PROJECT_PATH}/requirements.txt"

    # Register this venv as a Jupyter kernel, so that Jupyter or VS Code launched from
    # anywhere can run the notebook. PYTHONPATH is baked into the kernelspec, so
    # `import ioi_hydra` works even if this file wasn't sourced.
    python -m ipykernel install --user --name ${KERNEL_NAME} --display-name "Python (${KERNEL_NAME})" \
        --env PYTHONPATH "${PROJECT_PATH}/python"
fi


##-----------------------------------------------------------------------------
echo "setup.sh done"
