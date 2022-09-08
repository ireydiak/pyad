$config_root = $args[0]
$data_fname = $args[1]
$trainer_fname = $args[2]
Set-Location -Path ..\
$items = Get-ChildItem -Path $config_root -Exclude "_*"

foreach ($item in $items) {

    Try {

        python main.py --config=$config_root\$data_fname --config=$config_root\$trainer_fname --config=$item

    } Catch {

        Write-Host("fatal error encountered for " + $item.Name)

    }

}