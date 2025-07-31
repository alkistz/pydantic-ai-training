from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_embedding_endpoint: str
    azure_openai_api_version: str = "2023-05-15"
    


settings = Settings()


print(settings)
