import asyncio

from autoagora.misc import async_exit_on_exception


class TestMiscFn:
    def test_exit_on_exception(self):
        exit_code = 500

        @async_exit_on_exception(exit_code=exit_code)
        async def async_function_with_exception():
            raise ValueError("Something went wrong")

        @async_exit_on_exception(exit_code=exit_code)
        async def async_function_no_exception():
            return 200

        try:
            asyncio.run(async_function_with_exception())
        except SystemExit as e:
            assert e.code == exit_code

        result = asyncio.run(async_function_no_exception())
        assert result == 200
