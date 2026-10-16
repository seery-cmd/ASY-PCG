#!/bin/bash
function ergodic_run() {
targetDir=`ls $1`

  for file in $targetDir
    do
      if [ -d  $1"/"$file ]
      then
        ergodic_run  $1"/"$file
      else
        if [ ${file:0-4} == '.mtx' ]  ;
          then
            make p_run test_matrix=$1"/"$file;
      fi
  fi
  done

}

path_str=/home/xiaogq/Seery-experiment/datasets
ergodic_run $path_str
