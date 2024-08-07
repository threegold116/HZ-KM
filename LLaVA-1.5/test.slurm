#!/bin/bash
#SBATCH --job-name="bench"
#SBATCH -o test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH -t 1:00:00                         # 任务运行的最长时间为 1 小时
#SBATCH --cpus-per-task=2                    # cpu数量，如果排不上队可以检查一下CPU数量够不够，命令： sinfo -N -o "%5N  %5t  %13C  %8O  %8e  %7m  %G"
                                              # 其中对应i的数量是空余数量
#SBATCH -w gpu08
#SBATCH --gres=gpu:a100-sxm4-80gb:1

/usr/sbin/sshd -D -p 7879 -f /dev/null -h ${HOME}/.ssh/id_rsa # uses the user key as the host key