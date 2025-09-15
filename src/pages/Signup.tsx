import { useNavigate } from "react-router-dom";
import { SignupForm } from "@/components/auth/SignupForm";

export default function Signup() {
  const navigate = useNavigate();

  const handleSignupSuccess = (userId: string) => {
    // Navigate to dashboard after successful signup
    navigate("/dashboard", { state: { userId } });
  };

  const handleCancel = () => {
    // Navigate back to home
    navigate("/");
  };

  return (
    <SignupForm 
      onSignupSuccess={handleSignupSuccess}
      onCancel={handleCancel}
    />
  );
}