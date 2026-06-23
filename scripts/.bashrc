
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/gpfs/software/hali/mamba/25.3.1-0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/gpfs/software/hali/mamba/25.3.1-0/etc/profile.d/conda.sh" ]; then
        . "/gpfs/software/hali/mamba/25.3.1-0/etc/profile.d/conda.sh"
    else
        export PATH="/gpfs/software/hali/mamba/25.3.1-0/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

