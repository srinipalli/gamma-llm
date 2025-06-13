# Set your new IP address
new_ip = '192.168.1.400'

with open(r'D:\Innova\Infra\Data\Stage\server1\cpu.log', 'r') as infile, \
     open('D:\Innova\Infra\Data\Stage\server1\cpu1.log', 'w') as outfile:
    for line in infile:
        parts = line.rstrip('\n').split(',')
        # Only replace if there are at least 10 fields
        if len(parts) > 9:
            parts[9] = new_ip  # Replace the IP address field
        outfile.write(','.join(parts) + '\n')
