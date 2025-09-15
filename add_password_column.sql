-- Add Password column to MUFG table
-- Run this in your Supabase SQL Editor

ALTER TABLE public."MUFG" 
ADD COLUMN IF NOT EXISTS "Password" TEXT;

-- Add comment to the column
COMMENT ON COLUMN public."MUFG"."Password" IS 'User password for login authentication';

-- Optional: Add a constraint to ensure password is not empty
ALTER TABLE public."MUFG" 
ADD CONSTRAINT password_not_empty 
CHECK (LENGTH("Password") >= 6);
