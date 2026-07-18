async function deleteUser(userId) {
    if (!confirm("Delete this user? Their encrypted embedding will be permanently removed.")) return;

    const res = await fetch(`/api/users/${userId}`, { method: "DELETE" });
    const data = await res.json();

    if (data.success) {
        const row = document.querySelector(`tr[data-user-id="${userId}"]`);
        if (row) row.remove();
    } else {
        alert("Failed to delete user.");
    }
}
