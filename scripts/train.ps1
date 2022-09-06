$config_root = $args[0]
Set-Location -Path ..\
$items = Get-ChildItem -Path $config_root -Exclude "_*"

foreach ($item in $items) {

    Try {

        python main.py --config=$config_root\_data.yaml --config=$config_root\_trainer.yaml --config=$item

    } Catch {

        Write-Host("fatal error encountered for " + $item.Name)

    }

}