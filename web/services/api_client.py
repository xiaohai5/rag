from __future__ import annotations

from typing import Any

import requests

from config import API_BASE_URL


class ApiClient:
    def __init__(self, base_url: str = API_BASE_URL, timeout: int = 30, chat_timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.chat_timeout = chat_timeout

    def register(self, username: str, email: str, password: str) -> dict[str, Any]:
        return self._post(
            "/auth/register",
            {"username": username, "email": email, "password": password},
        )

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._post("/auth/login", {"username": username, "password": password})

    def get_profile(self) -> dict[str, Any]:
        return self._get("/auth/profile", auth=True)

    def change_password(
        self,
        username: str,
        old_password: str,
        new_password: str,
        confirm_password: str,
    ) -> dict[str, Any]:
        return self._post(
            "/auth/change-password",
            {
                "username": username,
                "old_password": old_password,
                "new_password": new_password,
                "confirm_password": confirm_password,
            },
            auth=True,
        )

    def upload_document(self, file_name: str, file_bytes: bytes) -> dict[str, Any]:
        files = {"file": (file_name, file_bytes)}
        return self._post("/vector-store/upload", files=files, auth=True)

    def list_documents(self) -> list[dict[str, Any]]:
        return self._get("/vector-store/documents", auth=True)

    def chat(
        self,
        question: str,
        top_k: int,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        return self._post(
            "/chat/completion",
            {
                "question": question,
                "top_k": top_k,
                "history": history,
            },
            auth=True,
            timeout=self.chat_timeout,
        )

    def _get(self, path: str, auth: bool = False, timeout: int | None = None) -> Any:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                timeout=timeout or self.timeout,
                headers=self._build_headers(auth=auth),
            )
        except requests.exceptions.ReadTimeout as exc:
            raise RuntimeError("Request timed out. Please retry or increase timeout.") from exc
        return self._handle_response(response)

    def _post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        auth: bool = False,
        timeout: int | None = None,
    ) -> Any:
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                json=json,
                data=data,
                files=files,
                timeout=timeout or self.timeout,
                headers=self._build_headers(auth=auth),
            )
        except requests.exceptions.ReadTimeout as exc:
            raise RuntimeError("Request timed out. Please retry or increase timeout.") from exc
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response: requests.Response) -> Any:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Server response is not valid JSON") from exc

        if not response.ok:
            detail = payload.get("detail", "Request failed")
            raise RuntimeError(detail)
        return payload

    @staticmethod
    def _build_headers(auth: bool = False) -> dict[str, str]:
        if not auth:
            return {}

        try:
            import streamlit as st
        except ImportError:
            return {}

        token = st.session_state.get("token", "")
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}


api_client = ApiClient()
