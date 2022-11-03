from autoagora.indexer_utils import hex_to_ipfs_hash, ipfs_hash_to_hex


class TestHexIpfs:
    def test_hex_ipfs(self):
        hex = "0xbbde25a2c85f55b53b7698b9476610c3d1202d88870e66502ab0076b7218f98a"
        ipfs = hex_to_ipfs_hash(hex)
        assert ipfs == "Qmaz1R8vcv9v3gUfksqiS9JUz7K9G8S5By3JYn8kTiiP5K"

    def test_ipfs_hex(self):
        ipfs = "Qmaz1R8vcv9v3gUfksqiS9JUz7K9G8S5By3JYn8kTiiP5K"
        hex = ipfs_hash_to_hex(ipfs)
        assert (
            hex == "0xbbde25a2c85f55b53b7698b9476610c3d1202d88870e66502ab0076b7218f98a"
        )
