# Get the WSL network adapter name
$wslAdapter = Get-NetAdapter | Where-Object {$_.InterfaceAlias -like "vEthernet (WSL*)"} | Select-Object -ExpandProperty InterfaceAlias

if ($wslAdapter) {
    Write-Output "WSL Network Adapter Found: $wslAdapter"

    # Check if a firewall rule already exists for the adapter
    $existingRule = Get-NetFirewallRule | Where-Object {
        $_.DisplayName -eq "Allow WSL Inbound" -and $_.Enabled -eq "True"
    }

    if ($existingRule) {
        Write-Output "Firewall rule already exists for WSL. No action needed."
    } else {
        # Create a new firewall rule to allow inbound connections for WSL
        New-NetFirewallRule -Name "WSLInbound" `
                            -DisplayName "Allow WSL Inbound" `
                            -Direction Inbound `
                            -InterfaceAlias $wslAdapter `
                            -Action Allow

        Write-Output "Firewall rule created successfully for WSL."
    }
} else {
    Write-Error "No WSL network adapter found. Ensure WSL is running and try again."
}

# Run PowerShell as Administrator:
# Press Win + X and select Terminal (Admin) or Windows PowerShell (Admin).

# Run the Script:
# Navigate to the directory where you saved the script, and execute it:


# Verify changes
# Open the Windows Defender Firewall panel or use the Get-NetFirewallRule command to confirm that the rule has been added.
