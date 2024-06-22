<?php
$pythonScript = "test_wc2.py";
$output = shell_exec("python $pythonScript 2>&1");
echo $output;
?>
