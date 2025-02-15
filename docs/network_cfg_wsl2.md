# Network Configuration for WSL Virtual Machine

# Enable Inbound Connection
1. Press `WIN+X`, then press `A` to open a terminal (administrator)
2. Execute the following PowerShell command:
```powershell
New-NetFirewallRule -Name WSLAllowAllInbound -DisplayName "WSL Allow All Inbound" -Direction Inbound -InterfaceAlias "vEthernet (WSL (Hyper-V firewall))" -Action Allow
```
Note: Use `ipconfig` to check the interface name first. The InterfaceAlias might be "vEthernet (WSL)" in earlier versions.

## Unity Editor Firewall Settings
If inbound connection to Unity editor is blocked, disable the inbound rules:
`Set-NetFirewallRule -DisplayName 'Unity 6000.0.32f1 Editor' -Enabled False`

## Restore System Changes
To revert the changes made above:
```
Remove-NetFirewallRule -Name WSLAllowAllInbound
Set-NetFirewallRule -DisplayName 'Unity 6000.0.32f1 Editor' -Enabled True
```

# Architecture
Both Simsreal and Simulator each run a publisher as a server, and then each run a subscriber as a client to connect to the other's publisher.

## Check IP Addresses
- On Windows Side: `ipconfig`, e.g.:
 ```
Windows IP 配置


以太网适配器 vEthernet (WSL (Hyper-V firewall)):

   连接特定的 DNS 后缀 . . . . . . . :
   本地链接 IPv6 地址. . . . . . . . : fe80::6aa:bbd2:76ec:a5b0%54
   IPv4 地址 . . . . . . . . . . . . : 172.27.160.1
   子网掩码  . . . . . . . . . . . . : 255.255.240.0
   默认网关. . . . . . . . . . . . . :

以太网适配器 以太网 2:

   连接特定的 DNS 后缀 . . . . . . . :
   自动配置 IPv4 地址  . . . . . . . : 169.254.107.44
   子网掩码  . . . . . . . . . . . . : 255.255.0.0
   默认网关. . . . . . . . . . . . . :

以太网适配器 以太网 3:

   连接特定的 DNS 后缀 . . . . . . . :
   本地链接 IPv6 地址. . . . . . . . : fe80::9726:5412:1853:24d5%60
   IPv4 地址 . . . . . . . . . . . . : 172.20.10.2
   子网掩码  . . . . . . . . . . . . : 255.255.255.240
   默认网关. . . . . . . . . . . . . : 172.20.10.1
```
Here windows' ip is`172.20.10.2`

- On Linux (WSL) Side: `ifconfig`, e.g.:
```
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.27.160.28  netmask 255.255.240.0  broadcast 172.27.175.255
        inet6 fe80::215:5dff:fea6:b515  prefixlen 64  scopeid 0x20<link>
        ether 00:15:5d:a6:b5:15  txqueuelen 1000  (Ethernet)
        RX packets 51694  bytes 102549878 (102.5 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 8816  bytes 875317 (875.3 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 5497  bytes 328725 (328.7 KB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 5497  bytes 328725 (328.7 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```
Here WSL's ip is`172.27.160.28`

## Python(Simsreal)
- `mian.py`:
    ```
    url = f"{robot_sub_cfg['protocol']}://{robot_sub_cfg['ip']}:{robot_sub_cfg['port']}"  # type: ignore
    ```
    `{robot_sub_cfg['ip']}` substitue as the Host ip address

- `congif.ymal`:
    - `sub` & `pub`
    `0.0.0.0`listens to all addresses; sub acts as a client to connect to the server, fill in the peer's IP address.
        ```
        robot:
          sub:
            protocol: tcp
            ip: 0.0.0.0
            port: 5556
          pub:
            protocol: tcp
            ip: 172.27.160.28 <- WSL ip address
            port: 5557
          mjcf_path: /mnt/e/simsreal/simsreal/simulator/Assets/MJCF/humanoid.xml
        ```
    - `mjcf_path` need to change to the actual XML path imported into Unity. Use the pwd command to view the complete path.E.g.: `/mnt/e/simsreal/simsreal/simulator/Assets/MJCF/humanoid.xml`


## Unity Side (Simulator)
- `simulator\Assets\Scripts\Core\zmq_communicator.cs`
```
public ZmqCommunicator(
        string pubAddress = "tcp://0.0.0.0:5556",
        string subAddress = "tcp://172.27.160.28:5557") <- WSL ip address
```
